
import numpy as np, json, argparse
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report

parser = argparse.ArgumentParser()
parser.add_argument("--augmented", action="store_true",
                    help="Also train prompt-augmented router (requires P1-P5 train npy)")
args = parser.parse_args()

ACT_DIR     = Path("activations")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)
TEMPLATES   = ["P0", "P1", "P2", "P3", "P4", "P5"]
MODES       = ["answer", "refuse", "conflict"]

with open(RESULTS_DIR / "layer_probe_hidden.json") as f:
    layer_results = json.load(f)
best_layer = int(max(layer_results, key=lambda k: layer_results[k]["val_acc"]))
print(f"Using best_layer={best_layer}")

# ── Router trained on P0 train only ──────────────────────────────────────────
H_tr = np.load(ACT_DIR / "H_P0_train.npy").astype(np.float32)[:, best_layer, :]
y_tr = np.load(ACT_DIR / "y_P0_train.npy")

scaler_p0 = StandardScaler().fit(H_tr)
clf_p0    = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                                solver="lbfgs", random_state=42)
clf_p0.fit(scaler_p0.transform(H_tr), y_tr)
print("Router trained on P0 train.\n")

results = {}
print("Train-on-P0 → test-on-template:")
for tmpl in TEMPLATES:
    hf = ACT_DIR / f"H_{tmpl}_val.npy"
    yf = ACT_DIR / f"y_{tmpl}_val.npy"
    if not hf.exists():
        print(f"  {tmpl}: val activations not found — skipping "
              f"(run: python 2_extract_allayer.py --template {tmpl} --split val)")
        results[tmpl] = {"status": "missing"}
        continue

    H_val = np.load(hf).astype(np.float32)[:, best_layer, :]
    y_val = np.load(yf)

    # Use P0 scaler — intentional: tests out-of-distribution template robustness
    pred  = clf_p0.predict(scaler_p0.transform(H_val))
    acc   = float(clf_p0.score(scaler_p0.transform(H_val), y_val))
    f1    = float(f1_score(y_val, pred, average="macro"))

    results[tmpl] = {"val_acc": acc, "val_f1": f1, "n": int(len(y_val))}
    print(f"  {tmpl}: val_acc={acc:.4f}  val_f1={f1:.4f}  n={len(y_val)}")
    print(classification_report(y_val, pred, target_names=MODES, digits=3))

# ── Optional: prompt-augmented router ────────────────────────────────────────
# Fix 5: only runs if --augmented flag passed AND train npy files exist
if args.augmented:
    aug_H, aug_y = [H_tr], [y_tr]   # always include P0
    found_extra = False
    for tmpl in ["P1", "P2", "P3", "P4", "P5"]:
        hf = ACT_DIR / f"H_{tmpl}_train.npy"
        yf = ACT_DIR / f"y_{tmpl}_train.npy"
        if hf.exists():
            aug_H.append(np.load(hf).astype(np.float32)[:, best_layer, :])
            aug_y.append(np.load(yf))
            found_extra = True
            print(f"  Loaded {tmpl} train activations for augmentation")
        else:
            print(f"  {tmpl} train npy not found — not included in augmented set")

    if not found_extra:
        print("⚠️  No extra template train activations found. "
              "Run 2_extract_allayer.py for P1–P5 --split train first.")
    else:
        aug_H      = np.vstack(aug_H)
        aug_y      = np.concatenate(aug_y)
        scaler_aug = StandardScaler().fit(aug_H)
        clf_aug    = LogisticRegression(max_iter=1000, C=1.0,
                                         multi_class="multinomial",
                                         solver="lbfgs", random_state=42)
        clf_aug.fit(scaler_aug.transform(aug_H), aug_y)
        print("\nPrompt-augmented router:")
        for tmpl in TEMPLATES:
            hf = ACT_DIR / f"H_{tmpl}_val.npy"
            yf = ACT_DIR / f"y_{tmpl}_val.npy"
            if not hf.exists():
                continue
            H_val = np.load(hf).astype(np.float32)[:, best_layer, :]
            y_val = np.load(yf)
            pred  = clf_aug.predict(scaler_aug.transform(H_val))
            acc   = float(clf_aug.score(scaler_aug.transform(H_val), y_val))
            f1    = float(f1_score(y_val, pred, average="macro"))
            results[f"{tmpl}_augmented"] = {"val_acc": acc, "val_f1": f1}
            print(f"  {tmpl} (aug): val_acc={acc:.4f}  val_f1={f1:.4f}")

out_file = RESULTS_DIR / "prompt_dependence.json"
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved → {out_file}")
