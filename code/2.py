
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy as scipy_entropy

from prompts import render_prompt

# Optional: if prompts.py defines this, use it
try:
    from prompts import MAX_DOC_TOKENS
except Exception:
    MAX_DOC_TOKENS = 256


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()

# Original script args
parser.add_argument("--template", default="P0")
parser.add_argument("--split", default="train")
parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
parser.add_argument(
    "--batch_clean",
    type=int,
    default=10,
    help="Call empty_cache/gc every N examples.",
)
parser.add_argument(
    "--save_dtype",
    choices=["float16", "float32"],
    default="float16",
    help="Storage dtype for saved numpy arrays H/M.",
)

# SAD-related args
parser.add_argument(
    "--out_json",
    default="sad_features_v2_alllayers.json",
    help="Keep same default output filename/location as previous SAD script.",
)
parser.add_argument(
    "--max_doc_tokens",
    type=int,
    default=MAX_DOC_TOKENS,
    help="Max doc tokens for boundary matching / SAD aggregation. "
         "Docs are matched in prompt using truncated tokenized doc spans.",
)
parser.add_argument(
    "--ckpt_every",
    type=int,
    default=200,
    help="Checkpoint SAD JSON every N examples.",
)
parser.add_argument(
    "--write_sad_json",
    action="store_true",
    help="If set, write/update sad_features_v2_alllayers.json. "
         "Default behavior is ON; this flag is here only for clarity.",
)
parser.add_argument(
    "--no_write_sad_json",
    action="store_true",
    help="Disable writing the SAD JSON file.",
)

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = Path("activations")
OUT_DIR.mkdir(exist_ok=True)

SAVE_DTYPE = np.float16 if args.save_dtype == "float16" else np.float32
WRITE_SAD_JSON = not args.no_write_sad_json


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_layers(m):
    """
    Return list of decoder blocks for common architectures.
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
    """
    if hasattr(block, "mlp"):
        return block.mlp
    if hasattr(block, "ffn"):
        return block.ffn
    if hasattr(block, "feed_forward"):
        return block.feed_forward
    raise AttributeError(
        f"Cannot find MLP/FFN module in block type {type(block)}. "
        "Add a new branch in get_mlp()."
    )


def get_self_attn(block):
    """
    Return the self-attention module inside a decoder block.
    """
    if hasattr(block, "self_attn"):
        return block.self_attn
    if hasattr(block, "attn"):
        return block.attn
    if hasattr(block, "attention"):
        return block.attention
    raise AttributeError(
        f"Cannot find self-attn module in block type {type(block)}. "
        "Add a new branch in get_self_attn()."
    )


def softmax_np(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x = x.astype(np.float32)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def find_subsequence(full_ids, sub_ids, start_pos=0):
    """
    Return first index i >= start_pos such that
      full_ids[i : i+len(sub_ids)] == sub_ids
    else return -1.
    """
    n = len(full_ids)
    m = len(sub_ids)
    if m == 0 or m > n:
        return -1
    last = n - m
    for i in range(start_pos, last + 1):
        if full_ids[i:i + m] == sub_ids:
            return i
    return -1


def recover_doc_token_bounds_from_rendered_prompt(
    tokenizer,
    prompt_text: str,
    docs: list,
    max_doc_tokens: int,
):
    """
    Recover token spans for each doc by tokenizing the FULL rendered prompt
    and matching each tokenized doc (possibly truncated to max_doc_tokens)
    as a subsequence inside the full prompt token ids, in order.

    This preserves the first script's prompt exactly.

    Returns:
      input_ids: torch.LongTensor [1, seq]
      seg_bounds: list of (start, end) token indices for each doc in input_ids
    """
    full_ids = tokenizer(prompt_text, add_special_tokens=True).input_ids
    seg_bounds = []
    cursor = 0

    for doc in docs:
        doc_ids = tokenizer(doc, add_special_tokens=False).input_ids
        if len(doc_ids) > max_doc_tokens:
            doc_ids = doc_ids[:max_doc_tokens]

        pos = find_subsequence(full_ids, doc_ids, start_pos=cursor)

        if pos == -1:
            # Fall back to empty span if not found; SAD features will safely handle it.
            seg_bounds.append((0, 0))
        else:
            seg_bounds.append((pos, pos + len(doc_ids)))
            cursor = pos + len(doc_ids)

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    return input_ids, seg_bounds


def sad_features_from_attn_row(seg_bounds, retriever_scores, attn_row_heads_seq: torch.Tensor):
    """
    attn_row_heads_seq: [heads, seq_len] attention from LAST QUERY token to all keys.
    Compute 7 SAD features.
    """
    n_segs = len(seg_bounds)
    if n_segs == 0:
        return [0.0] * 7

    sl = attn_row_heads_seq.shape[1]
    A = np.zeros(n_segs, dtype=np.float32)

    for sidx, (s, e) in enumerate(seg_bounds):
        if s >= sl:
            continue
        e = min(e, sl)
        if e <= s:
            continue
        # mean over heads, sum over doc positions
        A[sidx] = attn_row_heads_seq[:, s:e].sum(dim=-1).mean().item()

    A_sum = float(A.sum())
    if A_sum > 0:
        A_norm = A / A_sum
    else:
        A_norm = np.ones(n_segs, dtype=np.float32) / float(n_segs)

    if retriever_scores is not None:
        R = softmax_np(np.array(retriever_scores, dtype=np.float32))
    else:
        R = np.ones(n_segs, dtype=np.float32) / float(n_segs)

    if R.size != n_segs:
        R = np.ones(n_segs, dtype=np.float32) / float(n_segs)

    delta = A_norm - R
    rsort = sorted(list(retriever_scores), reverse=True) if retriever_scores is not None else []
    top1r = float(rsort[0]) if len(rsort) >= 1 else 0.0
    top2r = float(rsort[1]) if len(rsort) >= 2 else 0.0

    return [
        float(delta.max()),                  # max mismatch
        float(scipy_entropy(A_norm + 1e-9)), # attention entropy over docs
        float(A_norm.max()),                 # max attention mass
        int(np.argmax(A_norm)),              # argmax doc index
        top1r,
        top2r,
        float(top1r - top2r),
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Load instances
# ──────────────────────────────────────────────────────────────────────────────
instances = []
with open(f"instances_{args.split}.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        instances.append(json.loads(line))

print(
    f"[{time.strftime('%H:%M:%S')}] Loaded {len(instances)} instances "
    f"({args.split}/{args.template})"
)

N = len(instances)


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
# Model metadata
# ──────────────────────────────────────────────────────────────────────────────
layers = get_layers(model)
L = model.config.num_hidden_layers
D = model.config.hidden_size

if len(layers) != L:
    print(
        f"⚠️  Warning: model.config.num_hidden_layers={L} but len(layers)={len(layers)}. "
        "Proceeding with len(layers)."
    )
    L = len(layers)

probe_layers = list(range(L))
print(f"Model: L={L} transformer layers, D={D}, N={N}")


# ──────────────────────────────────────────────────────────────────────────────
# MLP hooks (do not modify forward outputs)
# ──────────────────────────────────────────────────────────────────────────────
mlp_cache = {}

def make_mlp_hook(layer_idx):
    def hook(module, _input, output):
        raw = output[0] if isinstance(output, tuple) else output
        if raw.ndim != 3:
            raise RuntimeError(f"Unexpected MLP output shape at L{layer_idx}: {tuple(raw.shape)}")

        # Store CPU float32 copy only for analysis; do not mutate 'raw'
        mlp_cache[layer_idx] = raw[0, -1, :].detach().cpu().float()
        return output
    return hook


mlp_hooks = []
for l in range(L):
    mlp_mod = get_mlp(layers[l])
    mlp_hooks.append(mlp_mod.register_forward_hook(make_mlp_hook(l)))

print(f"[{time.strftime('%H:%M:%S')}] Registered MLP hooks on {len(mlp_hooks)} layers")


# ──────────────────────────────────────────────────────────────────────────────
# Optional attention hooks (fallback only)
# We mainly use out.attentions, but keep hooks as a backup for some architectures.
# ──────────────────────────────────────────────────────────────────────────────
attn_cache = {}

def make_attn_hook(layer_idx):
    def hook(module, _input, output):
        try:
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                attn = output[1]  # expected [B, heads, q, k]
                if attn.ndim == 4:
                    attn_cache[layer_idx] = attn[0, :, -1, :].detach().cpu().float()
        except Exception:
            pass
        return output
    return hook


attn_hooks = []
for l in range(L):
    attn_mod = get_self_attn(layers[l])
    attn_hooks.append(attn_mod.register_forward_hook(make_attn_hook(l)))

print(f"[{time.strftime('%H:%M:%S')}] Registered attention hooks on {len(attn_hooks)} layers")


# ──────────────────────────────────────────────────────────────────────────────
# Allocate outputs from first script (unchanged)
# H: [N, L, D] transformer layer outputs (embedding excluded)
# M: [N, L, D] MLP outputs
# ──────────────────────────────────────────────────────────────────────────────
H = np.zeros((N, L, D), dtype=SAVE_DTYPE)
M = np.zeros((N, L, D), dtype=SAVE_DTYPE)
labels = np.zeros(N, dtype=np.int8)
meta = []

# For SAD JSON rows
sad_rows = []
sad_errors = 0
out_path = Path(args.out_json)
if out_path.parent != Path(""):
    out_path.parent.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main extraction loop
# ──────────────────────────────────────────────────────────────────────────────
for idx, inst in enumerate(tqdm(instances, desc=f"{args.template}/{args.split}")):
    try:
        prompt = render_prompt(args.template, inst["question"], inst["docs"], tokenizer=tokenizer)

        # Preserve original prompt from first script, but recover doc spans for SAD
        input_ids, seg_bounds = recover_doc_token_bounds_from_rendered_prompt(
            tokenizer=tokenizer,
            prompt_text=prompt,
            docs=inst["docs"],
            max_doc_tokens=args.max_doc_tokens,
        )
        enc = {"input_ids": input_ids.to(DEVICE)}

        mlp_cache.clear()
        attn_cache.clear()

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, output_attentions=True)

        # ── H from hidden_states (same as first script)
        hs = out.hidden_states  # tuple length L+1 (0=embeds)
        for layer_idx in range(L):
            vec = hs[layer_idx + 1][0, -1, :].detach().cpu().float().numpy()
            H[idx, layer_idx, :] = vec.astype(SAVE_DTYPE, copy=False)

        # ── M from MLP hooks (same as first script)
        for layer_idx in range(L):
            if layer_idx in mlp_cache:
                M[idx, layer_idx, :] = mlp_cache[layer_idx].numpy().astype(SAVE_DTYPE, copy=False)

        # ── Labels/meta (same as first script)
        labels[idx] = inst["label"]
        meta.append(
            {
                "id": inst["id"],
                "original_id": inst["original_id"],
                "true_mode": inst["true_mode"],
            }
        )

        # ── SAD extraction
        retr_scores = inst.get("retriever_scores", None)

        # Prefer out.attentions if available; fallback to attn_cache if needed
        present_rows = []
        feats_by_layer = []

        attns = getattr(out, "attentions", None)

        for layer_idx in probe_layers:
            row = None

            if attns is not None and len(attns) > layer_idx and attns[layer_idx] is not None:
                # attns[layer_idx]: [B, heads, q, k]
                a = attns[layer_idx]
                if a.ndim == 4:
                    row = a[0, :, -1, :].detach().cpu().float()

            if row is None and layer_idx in attn_cache:
                row = attn_cache[layer_idx]

            if row is not None:
                feats = sad_features_from_attn_row(seg_bounds, retr_scores, row)
                feats_by_layer.append(feats)
                present_rows.append(row)
            else:
                feats_by_layer.append([0.0] * 7)

        feats_by_layer = np.array(feats_by_layer, dtype=np.float32)

        if len(present_rows) > 0:
            attn_mean = torch.stack(present_rows, dim=0).mean(dim=0)  # [heads, seq]
            feats_mean = sad_features_from_attn_row(seg_bounds, retr_scores, attn_mean)
        else:
            feats_mean = [0.0] * 7

    except Exception as ex:
        sad_errors += 1

        # Preserve array shapes and outputs
        labels[idx] = inst.get("label", 0)
        meta.append(
            {
                "id": inst.get("id"),
                "original_id": inst.get("original_id"),
                "true_mode": inst.get("true_mode"),
            }
        )

        feats_mean = [0.0] * 7
        feats_by_layer = np.zeros((L, 7), dtype=np.float32)

    # Append SAD row in same style as previous script
    sad_rows.append(
        {
            "id": inst.get("id"),
            "split": args.split,
            "true_mode": inst.get("true_mode"),
            "label": inst.get("label"),
            "probe_layers": probe_layers,
            "features_mean": feats_mean,
            "features_by_layer": feats_by_layer.tolist(),
        }
    )

    # Cleanup
    if "out" in locals():
        del out
    if "enc" in locals():
        del enc
    if "input_ids" in locals():
        del input_ids

    mlp_cache.clear()
    attn_cache.clear()

    if (idx + 1) % args.batch_clean == 0:
        torch.cuda.empty_cache()
        gc.collect()

    if WRITE_SAD_JSON and args.ckpt_every and (idx + 1) % args.ckpt_every == 0:
        ckpt = out_path.with_name(out_path.stem + f"_ckpt{idx+1}.json")
        with open(ckpt, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {
                        "model": args.model,
                        "probe_layers": probe_layers,
                        "template": args.template,
                        "split": args.split,
                    },
                    "rows": sad_rows,
                },
                f,
            )
        print(f"  Checkpoint -> {ckpt}")


# ──────────────────────────────────────────────────────────────────────────────
# Final cleanup
# ──────────────────────────────────────────────────────────────────────────────
torch.cuda.empty_cache()
gc.collect()

for hk in mlp_hooks:
    hk.remove()

for hk in attn_hooks:
    hk.remove()


# ──────────────────────────────────────────────────────────────────────────────
# Save FIRST script outputs exactly as before
# ──────────────────────────────────────────────────────────────────────────────
tag = f"{args.template}_{args.split}"
np.save(OUT_DIR / f"H_{tag}.npy", H)
np.save(OUT_DIR / f"M_{tag}.npy", M)
np.save(OUT_DIR / f"y_{tag}.npy", labels)
with open(OUT_DIR / f"meta_{tag}.json", "w", encoding="utf-8") as f:
    json.dump(meta, f)

print(f"[{time.strftime('%H:%M:%S')}] Saved H{H.shape}, M{M.shape}, y{labels.shape} → {OUT_DIR}")
print("CONVENTION: H[:,l] ↔ layers[l] output; H[:,l]=hidden_states[l+1][:,-1,:]. No +1 elsewhere.")


# ──────────────────────────────────────────────────────────────────────────────
# Save SECOND script output in same default filename/location
# ──────────────────────────────────────────────────────────────────────────────
if WRITE_SAD_JSON:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "model": args.model,
                    "probe_layers": probe_layers,
                    "template": args.template,
                    "split": args.split,
                },
                "rows": sad_rows,
            },
            f,
        )

    print(f"[{time.strftime('%H:%M:%S')}] Done. sad_errors={sad_errors}. Saved -> {out_path}")
else:
    print(f"[{time.strftime('%H:%M:%S')}] Done. sad_errors={sad_errors}. SAD JSON writing disabled.")
