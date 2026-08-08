import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import render_prompt

# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--template", default="P0")
parser.add_argument("--split", default="train")
parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument(
    "--batch_clean",
    type=int,
    default=10,
    help="Call empty_cache/gc every N examples (not every example).",
)
parser.add_argument(
    "--save_dtype",
    choices=["float16", "float32"],
    default="float16",
    help="Storage dtype for saved numpy arrays (float16 smaller; float32 more faithful).",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("activations")
OUT_DIR.mkdir(exist_ok=True)

SAVE_DTYPE = np.float16 if args.save_dtype == "float16" else np.float32

# ──────────────────────────────────────────────────────────────────────────────
# Load instances
# ──────────────────────────────────────────────────────────────────────────────
instances = []
with open(f"instances_{args.split}.jsonl") as f:
    for line in f:
        instances.append(json.loads(line))
print(
    f"[{time.strftime('%H:%M:%S')}] Loaded {len(instances)} instances "
    f"({args.split}/{args.template})"
)

# ──────────────────────────────────────────────────────────────────────────────
# Load model + tokenizer
# ──────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(args.model)

# Qwen3-safe: bf16 weights; fp16 can cause dtype mismatch in some projections
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager",
)
model.eval()
for p in model.parameters():
    p.requires_grad = False

param0 = next(model.parameters())
print(f"[{time.strftime('%H:%M:%S')}] Model loaded: dtype={param0.dtype} device={param0.device}")

# ──────────────────────────────────────────────────────────────────────────────
# Model-agnostic accessors
# ──────────────────────────────────────────────────────────────────────────────
def get_layers(m):
    """
    Return list of decoder blocks for common architectures.
    Extend with more branches as needed when swapping models.
    """
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers  # Llama/Mistral/Qwen2/Qwen3/Phi-3 style
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h  # GPT-2 style
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return m.model.decoder.layers  # OPT style
    raise AttributeError(
        f"Cannot find decoder layers for model type {type(m)}. "
        "Add a new branch in get_layers()."
    )

def get_mlp(block):
    """
    Return the MLP/FFN module inside a decoder block, if present.
    If missing, raise a clear error (some blocks may use different names).
    """
    if hasattr(block, "mlp"):
        return block.mlp
    # Add common alternatives if you swap to other architectures:
    if hasattr(block, "ffn"):
        return block.ffn
    if hasattr(block, "feed_forward"):
        return block.feed_forward
    raise AttributeError(
        f"Cannot find MLP/FFN module in block type {type(block)}. "
        "Add a new branch in get_mlp()."
    )

layers = get_layers(model)
L = model.config.num_hidden_layers
D = model.config.hidden_size
N = len(instances)

if len(layers) != L:
    print(
        f"⚠️  Warning: model.config.num_hidden_layers={L} but len(layers)={len(layers)}. "
        "Proceeding with len(layers)."
    )
    L = len(layers)

print(f"Model: L={L} transformer layers, D={D}, N={N}")

# ──────────────────────────────────────────────────────────────────────────────
# MLP hooks (do not modify forward outputs)
# ──────────────────────────────────────────────────────────────────────────────
mlp_cache = {}

def make_mlp_hook(layer_idx):
    def hook(module, _input, output):
        # output can be tensor or tuple
        raw = output[0] if isinstance(output, tuple) else output
        if raw.ndim != 3:
            raise RuntimeError(f"Unexpected MLP output shape at L{layer_idx}: {tuple(raw.shape)}")

        # Store CPU float32 copy only for analysis; do not mutate 'raw'
        mlp_cache[layer_idx] = raw[0, -1, :].detach().cpu().float()

        # Return output unchanged to preserve dtype/flow
        return output
    return hook

mlp_hooks = []
for l in range(L):
    mlp_mod = get_mlp(layers[l])
    mlp_hooks.append(mlp_mod.register_forward_hook(make_mlp_hook(l)))
print(f"[{time.strftime('%H:%M:%S')}] Registered MLP hooks on {len(mlp_hooks)} layers")

# ──────────────────────────────────────────────────────────────────────────────
# Allocate outputs
# H: [N, L, D] transformer layer outputs (embedding excluded)
# M: [N, L, D] MLP outputs
# ──────────────────────────────────────────────────────────────────────────────
H = np.zeros((N, L, D), dtype=SAVE_DTYPE)
M = np.zeros((N, L, D), dtype=SAVE_DTYPE)
labels = np.zeros(N, dtype=np.int8)
meta = []

# ──────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────────────
for idx, inst in enumerate(tqdm(instances, desc=f"{args.template}/{args.split}")):
    prompt = render_prompt(args.template, inst["question"], inst["docs"], tokenizer=tokenizer)
    enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    mlp_cache.clear()
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)

    hs = out.hidden_states  # tuple length L+1 (0=embeds)
    # Store per-layer last-token hidden state
    for layer_idx in range(L):
        vec = hs[layer_idx + 1][0, -1, :].detach().cpu().float().numpy()
        H[idx, layer_idx, :] = vec.astype(SAVE_DTYPE, copy=False)

    # Store per-layer last-token MLP outputs (from hooks)
    for layer_idx in range(L):
        if layer_idx in mlp_cache:
            M[idx, layer_idx, :] = mlp_cache[layer_idx].numpy().astype(SAVE_DTYPE, copy=False)

    labels[idx] = inst["label"]
    meta.append(
        {
            "id": inst["id"],
            "original_id": inst["original_id"],
            "true_mode": inst["true_mode"],
        }
    )

    # Cleanup
    del out, enc
    mlp_cache.clear()

    if (idx + 1) % args.batch_clean == 0:
        torch.cuda.empty_cache()
        gc.collect()

torch.cuda.empty_cache()
gc.collect()

# Remove hooks (good hygiene if script is imported / reused)
for hk in mlp_hooks:
    hk.remove()

# ──────────────────────────────────────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────────────────────────────────────
tag = f"{args.template}_{args.split}"
np.save(OUT_DIR / f"H_{tag}.npy", H)
np.save(OUT_DIR / f"M_{tag}.npy", M)
np.save(OUT_DIR / f"y_{tag}.npy", labels)
with open(OUT_DIR / f"meta_{tag}.json", "w") as f:
    json.dump(meta, f)

print(f"[{time.strftime('%H:%M:%S')}] Saved H{H.shape}, M{M.shape}, y{labels.shape} → {OUT_DIR}")
print("CONVENTION: H[:,l] ↔ layers[l] output; H[:,l]=hidden_states[l+1][:,-1,:]. No +1 elsewhere.")
