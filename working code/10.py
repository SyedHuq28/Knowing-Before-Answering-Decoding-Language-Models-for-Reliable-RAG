import argparse, json, gc, time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.stats import entropy as scipy_entropy

# If you have this constant in prompts.py keep it. Otherwise set a safe default here.
try:
    from prompts import MAX_DOC_TOKENS
except Exception:
    MAX_DOC_TOKENS = 256


def get_decoder_layers(m):
    # Works for many HF decoder-only models (Llama/Qwen/Mistral/Granite-style)
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    raise AttributeError(f"Cannot find decoder layers for {type(m)}")


def get_self_attn(layer):
    # Common patterns
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise AttributeError(f"Cannot find self-attn module inside layer {type(layer)}")


def softmax_np(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    x = x.astype(np.float32)
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def build_prompt_with_boundaries(tokenizer, question: str, docs: list, max_doc_tokens: int):
    """
    Build a simple single-context prompt by token concatenation and return:
      input_ids: torch.LongTensor [1, seq]
      seg_bounds: list of (start,end) token indices for each doc in input_ids
    """
    BOS = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    NEWLINE = tokenizer("\n\n", add_special_tokens=False).input_ids

    system_text = "You are a helpful assistant. Use only the provided documents to answer."
    system_ids = tokenizer(system_text, add_special_tokens=False).input_ids

    q_prefix_ids = tokenizer("Question: ", add_special_tokens=False).input_ids
    q_ids = tokenizer(question, add_special_tokens=False).input_ids
    ans_ids = tokenizer("\nAnswer:", add_special_tokens=False).input_ids

    full_ids = BOS + system_ids + NEWLINE
    seg_bounds = []

    for i, doc in enumerate(docs):
        doc_tok = tokenizer(doc, add_special_tokens=False).input_ids
        if len(doc_tok) > max_doc_tokens:
            doc_tok = doc_tok[:max_doc_tokens]

        prefix_ids = tokenizer(f"[Document {i+1}]:\n", add_special_tokens=False).input_ids
        start = len(full_ids) + len(prefix_ids)
        end = start + len(doc_tok)
        seg_bounds.append((start, end))

        full_ids += prefix_ids + doc_tok + NEWLINE

    full_ids += q_prefix_ids + q_ids + ans_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    return input_ids, seg_bounds


def sad_features_from_attn_row(seg_bounds, retriever_scores, attn_row_heads_seq: torch.Tensor):
    """
    attn_row_heads_seq: [heads, seq_len] attention from LAST QUERY token to all keys.
    Compute 7 SAD features (same as your code) for one layer.
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

    R = softmax_np(np.array(retriever_scores, dtype=np.float32)) if retriever_scores is not None else np.ones(n_segs)/n_segs
    if R.size != n_segs:
        # if retriever_scores length mismatch, fall back safely
        R = np.ones(n_segs, dtype=np.float32) / float(n_segs)

    delta = A_norm - R
    rsort = sorted(list(retriever_scores), reverse=True) if retriever_scores is not None else []
    top1r = float(rsort[0]) if len(rsort) >= 1 else 0.0
    top2r = float(rsort[1]) if len(rsort) >= 2 else 0.0

    return [
        float(delta.max()),                          # max mismatch
        float(scipy_entropy(A_norm + 1e-9)),         # attention entropy over docs
        float(A_norm.max()),                         # max attention mass
        int(np.argmax(A_norm)),                      # argmax doc index
        top1r,
        top2r,
        float(top1r - top2r),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ibm-granite/granite-3.1-8b-instruct")
    ap.add_argument("--layers", nargs="+", default=["all"],
                    help='Layer indices, or "all". Example: --layers 0 1 2 OR --layers all')
    ap.add_argument("--max_doc_tokens", type=int, default=MAX_DOC_TOKENS)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--instances_pattern", default="instances_{split}.jsonl")
    ap.add_argument("--out_json", default="sad_features_v2_alllayers.json")
    ap.add_argument("--ckpt_every", type=int, default=200)
    ap.add_argument("--max_instances", type=int, default=None)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print(f"[{time.strftime('%H:%M:%S')}] Loading model={args.model} dtype={args.dtype} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        attn_implementation="eager",   # needed for attentions in many setups
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    layers = get_decoder_layers(model)
    n_layers_total = len(layers)
    print(f"[{time.strftime('%H:%M:%S')}] Model layers: {n_layers_total}")

    if len(args.layers) == 1 and args.layers[0].lower() == "all":
        probe_layers = list(range(n_layers_total))
    else:
        probe_layers = sorted({int(x) for x in args.layers})
        for l in probe_layers:
            if l < 0 or l >= n_layers_total:
                raise ValueError(f"Layer {l} out of range [0,{n_layers_total-1}]")

    print(f"[{time.strftime('%H:%M:%S')}] PROBE_LAYERS={probe_layers[:10]}{'...' if len(probe_layers)>10 else ''} (n={len(probe_layers)})")

    # --- Attention cache: store last-query row per layer: [heads, seq_len]
    attn_cache = {}

    def make_attn_hook(layer_idx):
        def hook(module, inp, out):
            # out: often (attn_output, attn_weights, ...) or similar
            if layer_idx not in probe_layers:
                return
            if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                # out[1]: [B, heads, q, k] OR [B, heads, seq, seq]
                attn = out[1]
                # store last query row -> [heads, seq_len]
                attn_cache[layer_idx] = attn[0, :, -1, :].detach().cpu().float()
        return hook

    hooks = []
    for li, layer in enumerate(layers):
        sa = get_self_attn(layer)
        hooks.append(sa.register_forward_hook(make_attn_hook(li)))
    print(f"[{time.strftime('%H:%M:%S')}] Registered {len(hooks)} attention hooks.")

    # --- Load instances
    all_instances = []
    for split in args.splits:
        path = args.instances_pattern.format(split=split)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                ex["split"] = ex.get("split", split)
                all_instances.append(ex)

    if args.max_instances is not None:
        all_instances = all_instances[:args.max_instances]

    print(f"[{time.strftime('%H:%M:%S')}] Instances: {len(all_instances)}")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    errors = 0

    for idx, inst in enumerate(tqdm(all_instances, desc="SAD all-layers")):
        try:
            q = inst["question"]
            docs = inst["docs"]
            retr_scores = inst.get("retriever_scores", None)

            input_ids, seg_bounds = build_prompt_with_boundaries(
                tokenizer, q, docs, args.max_doc_tokens
            )

            attn_cache.clear()
            with torch.no_grad():
                _ = model(input_ids=input_ids.to(device), output_attentions=True)

            # Per-layer features
            feats_by_layer = []
            for l in probe_layers:
                if l in attn_cache:
                    feats_by_layer.append(sad_features_from_attn_row(seg_bounds, retr_scores, attn_cache[l]))
                else:
                    feats_by_layer.append([0.0] * 7)
            feats_by_layer = np.array(feats_by_layer, dtype=np.float32)  # [n_probe_layers, 7]

            # Mean-across-layers (original style): average attention rows then compute features once
            # We average A-values implicitly by averaging attention rows first:
            present = [attn_cache[l] for l in probe_layers if l in attn_cache]
            if len(present) > 0:
                attn_mean = torch.stack(present, dim=0).mean(dim=0)  # [heads, seq]
                feats_mean = sad_features_from_attn_row(seg_bounds, retr_scores, attn_mean)
            else:
                feats_mean = [0.0] * 7

        except Exception as ex:
            errors += 1
            feats_mean = [0.0] * 7
            feats_by_layer = np.zeros((len(probe_layers), 7), dtype=np.float32)

        results.append({
            "id": inst.get("id"),
            "split": inst.get("split"),
            "true_mode": inst.get("true_mode"),
            "label": inst.get("label"),  # keep if you have it
            "probe_layers": probe_layers,
            "features_mean": feats_mean,                         # length 7
            "features_by_layer": feats_by_layer.tolist(),        # [n_layers, 7]
        })

        attn_cache.clear()
        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if args.ckpt_every and (idx + 1) % args.ckpt_every == 0:
            ckpt = out_path.with_name(out_path.stem + f"_ckpt{idx+1}.json")
            with open(ckpt, "w", encoding="utf-8") as f:
                json.dump(results, f)
            print(f"  Checkpoint -> {ckpt}")

    for h in hooks:
        h.remove()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": args.model, "probe_layers": probe_layers}, "rows": results}, f)

    print(f"[{time.strftime('%H:%M:%S')}] Done. errors={errors}. Saved -> {out_path}")


if __name__ == "__main__":
    main()
