import numpy as np, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

OUT_DIR     = Path("activations")
RESULTS_DIR = Path("results"); RESULTS_DIR.mkdir(exist_ok=True)


def load(template, split, space="H"):
    # H/M shape: [N, L, D] — transformer layers only, NO embedding row
    X = np.load(OUT_DIR / f"{space}_{template}_{split}.npy").astype(np.float32)
    y = np.load(OUT_DIR / f"y_{template}_{split}.npy")
    return X, y


def probe_layer(X_tr, y_tr, X_eval, y_eval):
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class="multinomial",
                              solver="lbfgs", random_state=42)
    clf.fit(scaler.fit_transform(X_tr), y_tr)
    pred    = clf.predict(scaler.transform(X_eval))
    val_acc = clf.score(scaler.transform(X_eval), y_eval)
    val_f1  = f1_score(y_eval, pred, average="macro")
    return val_acc, val_f1


X_tr,  y_tr  = load("P0", "train")
X_val, y_val = load("P0", "val")
X_te,  y_te  = load("P0", "test")

L = X_tr.shape[1]
print(f"Probing {L} transformer layers, d={X_tr.shape[2]}")

# ── Single-layer probes (hidden) ──────────────────────────────────────────────
results_h = {}
for l in range(L):
    val_acc, val_f1 = probe_layer(X_tr[:,l,:], y_tr, X_val[:,l,:], y_val)
    te_acc,  _      = probe_layer(X_tr[:,l,:], y_tr, X_te[:,l,:],  y_te)
    results_h[l] = {"val_acc": float(val_acc), "val_f1": float(val_f1),
                    "test_acc": float(te_acc)}
    print(f"H L{l:02d}: val_acc={val_acc:.4f}  val_f1={val_f1:.4f}  test_acc={te_acc:.4f}")

with open(RESULTS_DIR / "layer_probe_hidden.json", "w") as f:
    json.dump(results_h, f, indent=2)

# ── Single-layer probes (MLP) ─────────────────────────────────────────────────
X_tr_m,  _ = load("P0", "train", "M")
X_val_m, _ = load("P0", "val",   "M")
results_m = {}
for l in range(X_tr_m.shape[1]):
    val_acc, val_f1 = probe_layer(X_tr_m[:,l,:], y_tr, X_val_m[:,l,:], y_val)
    results_m[l] = {"val_acc": float(val_acc), "val_f1": float(val_f1)}
    print(f"M L{l:02d}: val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

with open(RESULTS_DIR / "layer_probe_mlp.json", "w") as f:
    json.dump(results_m, f, indent=2)

# ── Consecutive 2-layer band probes (hidden) ─────────────────────────────────
band_results = {}
for l in range(L - 1):
    h_tr_b  = X_tr[:,  l:l+2, :].reshape(len(X_tr),  -1)
    h_val_b = X_val[:, l:l+2, :].reshape(len(X_val), -1)
    val_acc, val_f1 = probe_layer(h_tr_b, y_tr, h_val_b, y_val)
    band_results[f"{l}-{l+1}"] = {"val_acc": float(val_acc), "val_f1": float(val_f1)}
    print(f"Band L{l}-{l+1}: val_acc={val_acc:.4f}")

with open(RESULTS_DIR / "band_probe_hidden.json", "w") as f:
    json.dump(band_results, f, indent=2)

best_layer = max(results_h, key=lambda k: results_h[k]["val_acc"])
print(f"\nBest single layer: L{best_layer} "
      f"(val_acc={results_h[best_layer]['val_acc']:.4f})")

