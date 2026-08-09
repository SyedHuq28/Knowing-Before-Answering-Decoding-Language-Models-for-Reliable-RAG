import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

COMPACT = "results/causal_tracing_val_compact.json"
OUT_DIR = Path("results")

DIRECTION_LABELS = {
    "refuse_to_answer":   "refuse → answer",
    "answer_to_refuse":   "answer → refuse",
    "conflict_to_answer": "conflict → answer",
    "answer_to_conflict": "answer → conflict",
}
COLORS = {
    "refuse_to_answer":   "#1f4e79",
    "answer_to_refuse":   "#2e75b6",
    "conflict_to_answer": "#2e8b57",
    "answer_to_conflict": "#70ad47",
}

with open(COMPACT) as f:
    data = json.load(f)

meta      = data["meta"]
results   = data["results"]
patch_l   = meta["patch_layer"]
obs_l     = meta["obs_layers"]
x         = np.array(obs_l)
directions = [d for d in DIRECTION_LABELS if d in results]

# ── Plot 1: Baseline vs Patched — one subplot per direction ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True, sharex=True)
axes = axes.flatten()

for i, d in enumerate(directions):
    ax    = axes[i]
    base  = np.array(results[d]["baseline_source_rate"])
    patch = np.array(results[d]["patched_source_rate"])
    color = COLORS[d]

    ax.plot(x, base,  marker="s", markersize=5, linestyle="--",
            color=color, alpha=0.5, linewidth=1.5, label="Baseline (no patch)")
    ax.plot(x, patch, marker="o", markersize=5, linestyle="-",
            color=color, linewidth=2,             label="Patched")

    # Shade the area between them
    ax.fill_between(x, base, patch,
                    where=(patch >= base),
                    alpha=0.12, color=color, label="Causal gain")

    ax.axvline(patch_l, color="red", linestyle="--",
               linewidth=1.5, label=f"Patch site L{patch_l}")
    ax.axhline(1/3, color="grey", linestyle=":", linewidth=1, label="Chance (0.33)")

    ax.set_title(DIRECTION_LABELS[d], fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}" for l in x], rotation=45, fontsize=7)
    ax.set_ylabel("Source-class prediction rate", fontsize=9)
    ax.set_xlabel("Observation layer", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

fig.suptitle(
    f"Causal Tracing: Baseline vs Patched Source Rate  (patch @ L{patch_l})",
    fontsize=14, fontweight="bold"
)
fig.tight_layout()
fig.savefig(OUT_DIR / "causal_tracing_baseline_vs_patched.png", dpi=150)
plt.close(fig)
print("Saved causal_tracing_baseline_vs_patched.png")


# ── Plot 2: Delta only — all directions on one axes ───────────────────────
fig, ax = plt.subplots(figsize=(11, 5))

for d in directions:
    delta = np.array(results[d]["delta"])
    ax.plot(x, delta, marker="o", markersize=5, linewidth=2,
            label=DIRECTION_LABELS[d], color=COLORS[d])

ax.axvline(patch_l, color="red", linestyle="--",
           linewidth=1.5, label=f"Patch site L{patch_l}")
ax.axhline(0.0,  color="grey",  linestyle=":",  linewidth=1)
ax.axhline(1/3,  color="orange",linestyle="--", linewidth=1,
           label="Chance delta (0.33)")

ax.set_xlabel("Observation Layer", fontsize=12)
ax.set_ylabel("Δ Source-class rate  (patched − baseline)", fontsize=12)
ax.set_title(
    f"Causal Propagation Signal (Δ) from Patch at L{patch_l}",
    fontsize=13, fontweight="bold"
)
ax.set_xticks(x)
ax.set_xticklabels([f"L{l}" for l in x], rotation=45, fontsize=8)
ax.set_ylim(-0.1, 1.0)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "causal_tracing_delta.png", dpi=150)
plt.close(fig)
print("Saved causal_tracing_delta.png")


# ── Plot 3: Side-by-side heatmaps (baseline | patched) ───────────────────
n_dirs = len(directions)
base_matrix  = np.zeros((n_dirs, len(obs_l)))
patch_matrix = np.zeros((n_dirs, len(obs_l)))
row_labels   = []

for i, d in enumerate(directions):
    base_matrix[i]  = np.array(results[d]["baseline_source_rate"])
    patch_matrix[i] = np.array(results[d]["patched_source_rate"])
    row_labels.append(DIRECTION_LABELS[d])

fig, (ax_b, ax_p) = plt.subplots(1, 2, figsize=(18, 3.5))
col_labels = [f"L{l}" for l in obs_l]
patch_col  = obs_l.index(patch_l)

for ax, matrix, title in [
    (ax_b, base_matrix,  "BASELINE (no patch)"),
    (ax_p, patch_matrix, "PATCHED"),
]:
    im = ax.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(obs_l)))
    ax.set_xticklabels(col_labels, rotation=45, fontsize=8)
    ax.set_yticks(range(n_dirs))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")

    for i in range(n_dirs):
        for j in range(len(obs_l)):
            val = matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if val > 0.55 else "black")

    # Red border on patch column
    for i in range(n_dirs):
        ax.add_patch(plt.Rectangle(
            (patch_col - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="red", linewidth=2
        ))

    plt.colorbar(im, ax=ax, label="Source-class rate")

fig.suptitle(
    f"Source-Class Prediction Rate Before vs After Patch  (patch site = L{patch_l})",
    fontsize=13, fontweight="bold"
)
fig.tight_layout()
fig.savefig(OUT_DIR / "causal_tracing_heatmap_compare.png", dpi=150)
plt.close(fig)
print("Saved causal_tracing_heatmap_compare.png")
