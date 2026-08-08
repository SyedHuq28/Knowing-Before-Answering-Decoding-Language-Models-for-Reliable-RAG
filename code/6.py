import argparse, json, pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV

MODES = ["answer", "refuse", "conflict"]

def pick_best_layer(layer_probe_path: Path) -> int:
    data = json.load(open(layer_probe_path, "r", encoding="utf-8"))
    # data keys are strings of layer idx
    best = max(data.items(), key=lambda kv: kv[1].get("val_acc", -1.0))[0]
    return int(best)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act_dir", default="activations")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--template", default="P0")
    ap.add_argument("--train_mode", choices=["train", "trainval"], default="train",
                    help="train=fit on train and calibrate on val (dev). trainval=fit on train+val (final).")
    ap.add_argument("--layer", type=int, default=None,
                    help="Override best layer. If omitted, uses results/layer_probe_hidden.json best val_acc.")
    ap.add_argument("--use_band2", type=int, default=0,
                    help="1 = use best 2-layer consecutive band from results/band_probe_hidden.json (flattened).")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--calibration", choices=["isotonic", "sigmoid", "none"], default="isotonic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    act_dir = Path(args.act_dir)
    res_dir = Path(args.results_dir)
    res_dir.mkdir(exist_ok=True)

    # Load labels
    y_tr  = np.load(act_dir / f"y_{args.template}_train.npy")
    y_val = np.load(act_dir / f"y_{args.template}_val.npy")
    y_te  = np.load(act_dir / f"y_{args.template}_test.npy")

    # Choose layer / band
    layer_probe = res_dir / "layer_probe_hidden.json"
    band_probe  = res_dir / "band_probe_hidden.json"

    if args.use_band2:
        if not band_probe.exists():
            raise FileNotFoundError(f"Missing {band_probe}. Run 3_layer_probe_sweep.py first.")
        band_data = json.load(open(band_probe, "r", encoding="utf-8"))
        best_band = max(band_data.items(), key=lambda kv: kv[1].get("val_acc", -1.0))[0]  # like "18-19"
        l0, l1 = [int(x) for x in best_band.split("-")]
        layers = [l0, l1]
        print(f"Using best 2-layer band: {best_band} (val_acc={band_data[best_band]['val_acc']:.4f})")
    else:
        if args.layer is not None:
            layers = [args.layer]
            print(f"Using overridden layer: L{layers[0]}")
        else:
            if not layer_probe.exists():
                raise FileNotFoundError(f"Missing {layer_probe}. Run 3_layer_probe_sweep.py first.")
            best_layer = pick_best_layer(layer_probe)
            lp = json.load(open(layer_probe, "r", encoding="utf-8"))
            print(f"Best layer from probe: L{best_layer} (val_acc={lp[str(best_layer)]['val_acc']:.4f})")
            layers = [best_layer]

    # Load hidden activations and slice
    H_tr  = np.load(act_dir / f"H_{args.template}_train.npy").astype(np.float32)
    H_val = np.load(act_dir / f"H_{args.template}_val.npy").astype(np.float32)
    H_te  = np.load(act_dir / f"H_{args.template}_test.npy").astype(np.float32)

    def featurize(H, layers):
        if len(layers) == 1:
            return H[:, layers[0], :]
        # band: flatten [N,2,D] -> [N,2D]
        return H[:, layers, :].reshape(H.shape[0], -1)

    X_tr  = featurize(H_tr, layers)
    X_val = featurize(H_val, layers)
    X_te  = featurize(H_te, layers)

    # Fit scaler on training data only (or train+val for final)
    if args.train_mode == "train":
        scaler = StandardScaler().fit(X_tr)
        Xtr_s  = scaler.transform(X_tr)
        Xval_s = scaler.transform(X_val)
        Xte_s  = scaler.transform(X_te)

        base = LogisticRegression(
            max_iter=2000, C=args.C, multi_class="multinomial",
            solver="lbfgs", random_state=args.seed
        )
        base.fit(Xtr_s, y_tr)

        if args.calibration == "none":
            router = base
        else:
            method = args.calibration
            # Calibrate on VAL only (no leakage into test)
            router = CalibratedClassifierCV(base, cv="prefit", method=method)
            router.fit(Xval_s, y_val)

        print("\nDEV Router (train, calibrated on val):")
        print("VAL report:")
        print(classification_report(y_val, router.predict(Xval_s), target_names=MODES))
        print("TEST report:")
        print(classification_report(y_te, router.predict(Xte_s), target_names=MODES))

        out_path = res_dir / "router_hidden_dev.pkl"

    else:
        # trainval final: fit scaler on train+val and calibrate with CV
        X_trainval = np.vstack([X_tr, X_val])
        y_trainval = np.concatenate([y_tr, y_val])

        scaler = StandardScaler().fit(X_trainval)
        Xtv_s  = scaler.transform(X_trainval)
        Xte_s  = scaler.transform(X_te)

        base = LogisticRegression(
            max_iter=2000, C=args.C, multi_class="multinomial",
            solver="lbfgs", random_state=args.seed
        )

        if args.calibration == "none":
            router = base.fit(Xtv_s, y_trainval)
        else:
            router = CalibratedClassifierCV(base, cv=3, method=args.calibration)
            router.fit(Xtv_s, y_trainval)

        print("\nFINAL Router (train+val):")
        print("TEST report:")
        print(classification_report(y_te, router.predict(Xte_s), target_names=MODES))

        out_path = res_dir / "router_hidden_final.pkl"

    payload = {
        "router": router,
        "scaler": scaler,
        "layers": layers,  # list: [best_layer] or [l0,l1]
        "feature_type": "hidden_last_token" if len(layers) == 1 else "hidden_band2_flat",
        "template": args.template,
        "train_mode": args.train_mode,
        "calibration": args.calibration,
        "C": args.C,
        "convention": (
            "Cached H arrays use: H[i, layer_idx] = hidden_states[layer_idx+1][0,-1,:]. "
            "When extracting online, use hidden_states[layer_idx+1]."
        ),
    }

    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved → {out_path}")

if __name__ == "__main__":
    main()
