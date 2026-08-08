import argparse, json
from pathlib import Path
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


LABELS = ["answer", "refuse", "conflict"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}


def load_rows(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return obj["meta"], obj["rows"]


def rows_to_arrays(rows, mode="mean"):
    """
    mode:
      - "mean": features_mean -> (N,7)
      - "all_concat": features_by_layer concat -> (N, 7*L)
      - "layer_k": features_by_layer[k] -> (N,7)
    """
    X, y, split = [], [], []
    for r in rows:
        tm = r.get("true_mode")
        if tm not in LABEL_TO_ID:
            continue
        if mode == "mean":
            feats = r["features_mean"]
        elif mode == "all_concat":
            feats = np.array(r["features_by_layer"], dtype=np.float32).reshape(-1).tolist()
        elif mode.startswith("layer_"):
            k = int(mode.split("_")[1])
            feats = r["features_by_layer"][k]
        else:
            raise ValueError(mode)

        X.append(feats)
        y.append(LABEL_TO_ID[tm])
        split.append(r.get("split"))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), np.array(split)


def eval_model(name, clf, Xtr, ytr, Xva, yva, Xte, yte):
    out = {}
    for tag, X, y in [("train", Xtr, ytr), ("val", Xva, yva), ("test", Xte, yte)]:
        pred = clf.predict(X)
        proba = clf.predict_proba(X) if hasattr(clf, "predict_proba") else None
        out[tag] = {
            "acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
        }
        if proba is not None:
            out[tag]["log_loss"] = float(log_loss(y, proba, labels=[0,1,2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sad_json", default="results/sad_features_v2_alllayers.json")
    ap.add_argument("--out_json", default="results/sad_models_summary.json")
    ap.add_argument("--use_mlp", type=int, default=0, help="Also train an MLP classifier on SAD features")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    meta, rows = load_rows(args.sad_json)
    probe_layers = meta.get("probe_layers", rows[0].get("probe_layers"))
    L = len(probe_layers)

    # Mean SAD
    X_mean, y, splits = rows_to_arrays(rows, mode="mean")
    # All-layer concat
    X_all, y2, splits2 = rows_to_arrays(rows, mode="all_concat")
    assert np.all(y == y2) and np.all(splits == splits2)

    def split_mask(s):
        return splits == s

    tr = split_mask("train")
    va = split_mask("val")
    te = split_mask("test")

    Xtr_m, Xva_m, Xte_m = X_mean[tr], X_mean[va], X_mean[te]
    ytr, yva, yte = y[tr], y[va], y[te]

    Xtr_a, Xva_a, Xte_a = X_all[tr], X_all[va], X_all[te]

    results = {
        "meta": {
            "sad_json": args.sad_json,
            "model": meta.get("model"),
            "probe_layers": probe_layers,
            "n_layers": L,
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "n_test": int(te.sum()),
        },
        "models": {}
    }

    # --- A) Logistic Regression on mean SAD
    sc_m = StandardScaler().fit(Xtr_m)
    clf_m = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial", random_state=args.seed)
    clf_m.fit(sc_m.transform(Xtr_m), ytr)
    results["models"]["LR_sad_mean"] = eval_model("LR_sad_mean", clf_m,
                                                 sc_m.transform(Xtr_m), ytr,
                                                 sc_m.transform(Xva_m), yva,
                                                 sc_m.transform(Xte_m), yte)

    # --- B) Logistic Regression on concat SAD
    sc_a = StandardScaler().fit(Xtr_a)
    clf_a = LogisticRegression(max_iter=4000, C=1.0, solver="lbfgs", multi_class="multinomial", random_state=args.seed)
    clf_a.fit(sc_a.transform(Xtr_a), ytr)
    results["models"]["LR_sad_all_concat"] = eval_model("LR_sad_all_concat", clf_a,
                                                       sc_a.transform(Xtr_a), ytr,
                                                       sc_a.transform(Xva_a), yva,
                                                       sc_a.transform(Xte_a), yte)

    # --- C) Best single-layer SAD (choose layer by val acc using LR)
    best = {"layer": None, "val_acc": -1.0, "val_f1": -1.0}
    per_layer = []
    for k in range(L):
        Xk, yk, sp = rows_to_arrays(rows, mode=f"layer_{k}")
        Xtr_k, Xva_k, Xte_k = Xk[tr], Xk[va], Xk[te]
        sc = StandardScaler().fit(Xtr_k)
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", multi_class="multinomial", random_state=args.seed)
        clf.fit(sc.transform(Xtr_k), ytr)
        pred_va = clf.predict(sc.transform(Xva_k))
        acc_va = float(accuracy_score(yva, pred_va))
        f1_va = float(f1_score(yva, pred_va, average="macro"))
        per_layer.append({"k": k, "layer": int(probe_layers[k]), "val_acc": acc_va, "val_f1": f1_va})

        if acc_va > best["val_acc"]:
            best = {"k": k, "layer": int(probe_layers[k]), "val_acc": acc_va, "val_f1": f1_va,
                    "scaler": sc, "clf": clf, "Xtr_k": Xtr_k, "Xva_k": Xva_k, "Xte_k": Xte_k}

    results["models"]["LR_sad_best_single_layer"] = {
        "best_layer": best["layer"],
        "best_k": best["k"],
        "val_acc": best["val_acc"],
        "val_f1": best["val_f1"],
        "per_layer": per_layer[:50],  # keep head; you can save full if you want
        "metrics": eval_model("LR_sad_best_single_layer", best["clf"],
                              best["scaler"].transform(best["Xtr_k"]), ytr,
                              best["scaler"].transform(best["Xva_k"]), yva,
                              best["scaler"].transform(best["Xte_k"]), yte)
    }

    # Optional MLPs (sometimes helps a bit)
    if args.use_mlp:
        mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=args.seed, early_stopping=True)
        mlp.fit(sc_a.transform(Xtr_a), ytr)
        results["models"]["MLP_sad_all_concat"] = eval_model("MLP_sad_all_concat", mlp,
                                                            sc_a.transform(Xtr_a), ytr,
                                                            sc_a.transform(Xva_a), yva,
                                                            sc_a.transform(Xte_a), yte)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {out_path}")

    # Print quick headline
    print("\n=== SAD baselines summary ===")
    for k, v in results["models"].items():
        if "metrics" in v:
            print(f"{k}: best_layer={v['best_layer']} val_acc={v['metrics']['val']['acc']:.4f} test_acc={v['metrics']['test']['acc']:.4f}")
        else:
            print(f"{k}: val_acc={v['val']['acc']:.4f} test_acc={v['test']['acc']:.4f}")


if __name__ == "__main__":
    main()
