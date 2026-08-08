import numpy as np, json
from pathlib import Path

OUT_DIR   = Path("activations")
STEER_DIR = Path("steering"); STEER_DIR.mkdir(exist_ok=True)

LABEL_MAP = {0: "answer", 1: "refuse", 2: "conflict"}


def compute_and_save(space: str, out_fname: str):
    X_tr = np.load(OUT_DIR / f"{space}_P0_train.npy").astype(np.float32)
    y_tr = np.load(OUT_DIR / f"y_P0_train.npy")
    L, D = X_tr.shape[1], X_tr.shape[2]

    means = {}
    for layer in range(L):
        means[layer] = {}
        for label, name in LABEL_MAP.items():
            mask = (y_tr == label)
            means[layer][name] = X_tr[mask, layer, :].mean(axis=0)

    # V[l, 0] = refuse vector;  V[l, 1] = conflict vector
    V = np.zeros((L, 2, D), dtype=np.float32)
    for layer in range(L):
        for m_idx, mode in enumerate(["refuse", "conflict"]):
            v = means[layer][mode] - means[layer]["answer"]
            V[layer, m_idx] = v / (np.linalg.norm(v) + 1e-8)

    np.save(STEER_DIR / out_fname, V)
    print(f"Saved {out_fname}: shape={V.shape}")

    diag = {
        layer: {
            "refuse_conflict_cosine":        float(np.dot(V[layer, 0], V[layer, 1])),
            "refuse_norm_raw":               float(np.linalg.norm(
                means[layer]["refuse"] - means[layer]["answer"])),
            "conflict_norm_raw":             float(np.linalg.norm(
                means[layer]["conflict"] - means[layer]["answer"])),
        }
        for layer in range(L)
    }
    diag_file = STEER_DIR / f"steering_diag_{space.lower()}.json"
    with open(diag_file, "w") as f:
        json.dump(diag, f, indent=2)
    print(f"Saved {diag_file}")


compute_and_save("H", "steering_vectors_hidden.npy")
compute_and_save("M", "steering_vectors_mlp.npy")

