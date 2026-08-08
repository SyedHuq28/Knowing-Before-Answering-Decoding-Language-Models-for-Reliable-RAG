import argparse, json, pickle, random, time, gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import render_prompt


LABEL_INV = {0: "answer", 1: "refuse", 2: "conflict"}
PATCH_DIRECTIONS = [
    ("refuse",   "answer"),
    ("answer",   "refuse"),
    ("conflict", "answer"),
    ("answer",   "conflict"),
]

# ──────────────────────────────────────────────────────────────────────────────
# Model helpers (device_map safe)
# ──────────────────────────────────────────────────────────────────────────────
def get_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return m.model.decoder.layers
    raise AttributeError(f"Cannot find decoder layers for {type(m)}")

def get_input_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for k in ["model.embed_tokens", "model.model.embed_tokens", "transformer.wte"]:
            if k in model.hf_device_map:
                return torch.device(model.hf_device_map[k])
        return torch.device(next(iter(model.hf_device_map.values())))
    return next(model.parameters()).device

def to_device(batch: Dict[str, torch.Tensor], dev: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(dev) for k, v in batch.items()}

def normalize_docs(docs):
    out = []
    for d in docs:
        if isinstance(d, str):
            s = d.strip()
        elif isinstance(d, dict):
            s = (d.get("text") or d.get("content") or d.get("contents") or "").strip()
        else:
            s = str(d).strip()
        if s:
            out.append(s)
    return out

# ──────────────────────────────────────────────────────────────────────────────
# Router helpers (support single-layer or 2-layer band)
# ──────────────────────────────────────────────────────────────────────────────
def route_from_hidden(router, scaler, feat_type: str, layers: List[int], hidden_states) -> Tuple[str, float]:
    # hidden_states is tuple len L+1; layer output is hidden_states[layer+1]
    if feat_type == "hidden_last_token":
        l = layers[0]
        vec = hidden_states[l + 1][0, -1, :].detach().cpu().float().numpy()
        X = vec.reshape(1, -1)
    elif feat_type == "hidden_band2_flat":
        l0, l1 = layers
        v0 = hidden_states[l0 + 1][0, -1, :].detach().cpu().float().numpy()
        v1 = hidden_states[l1 + 1][0, -1, :].detach().cpu().float().numpy()
        X = np.concatenate([v0, v1]).reshape(1, -1)
    else:
        raise ValueError(f"Unknown feature_type={feat_type}")

    Xs = scaler.transform(X)
    pred = int(router.predict(Xs)[0])
    proba = float(router.predict_proba(Xs).max())
    return LABEL_INV[pred], proba

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--router_pkl", required=True)
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--template", default="P0")
    ap.add_argument("--n_pairs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_file", default="results/causal_patching.json")
    ap.add_argument("--max_docs", type=int, default=None, help="Optional: truncate docs count for speed")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(exist_ok=True)

    # Load router bundle
    with open(args.router_pkl, "rb") as f:
        rdata = pickle.load(f)
    router = rdata["router"]
    scaler = rdata["scaler"]
    feat_type = rdata["feature_type"]
    router_layers = rdata["layers"]
    print(f"[{time.strftime('%H:%M:%S')}] Router={args.router_pkl} feat={feat_type} layers={router_layers}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    blocks = get_layers(model)
    input_dev = get_input_device(model)
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded. input_device={input_dev} L={len(blocks)}")

    # Load instances
    inst_path = Path(f"instances_{args.split}.jsonl")
    instances = {"answer": [], "refuse": [], "conflict": []}
    with open(inst_path, "r", encoding="utf-8") as f:
        for line in f:
            inst = json.loads(line)
            instances[inst["true_mode"]].append(inst)
    print({k: len(v) for k, v in instances.items()})

    # Hook store for injecting patched states: {layer_idx: tensor[D]}
    patch_store: Dict[int, torch.Tensor] = {}

    def make_inject_hook(layer_idx: int):
        def hook(_module, _inp, output):
            if layer_idx not in patch_store:
                return None
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output  # [B,T,D]
            if not torch.is_tensor(h) or h.ndim != 3:
                return None
            h2 = h.clone()
            h2[0, -1, :] = patch_store[layer_idx].to(h2.device).to(h2.dtype)
            if is_tuple:
                return (h2,) + output[1:]
            return h2
        return hook

    hooks = []
    for li, blk in enumerate(blocks):
        hooks.append(blk.register_forward_hook(make_inject_hook(li)))

    def forward_hidden(inst) -> Tuple[Tuple[torch.Tensor, ...], str, float]:
        docs = normalize_docs(inst["docs"])
        if args.max_docs is not None:
            docs = docs[:args.max_docs]
        prompt = render_prompt(args.template, inst["question"], docs, tokenizer=tokenizer)
        enc = to_device(tokenizer(prompt, return_tensors="pt"), input_dev)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        pred, conf = route_from_hidden(router, scaler, feat_type, router_layers, out.hidden_states)
        hs = out.hidden_states
        del enc, out
        return hs, pred, conf

    def pack_router_state(hs) -> Dict[int, np.ndarray]:
        # Return {layer_idx: np[D]} for each layer used by router
        if feat_type == "hidden_last_token":
            l = router_layers[0]
            return {l: hs[l + 1][0, -1, :].detach().cpu().float().numpy()}
        else:
            l0, l1 = router_layers
            return {
                l0: hs[l0 + 1][0, -1, :].detach().cpu().float().numpy(),
                l1: hs[l1 + 1][0, -1, :].detach().cpu().float().numpy(),
            }

    def route_from_forward(inst, patch: Dict[int, np.ndarray] = None) -> Tuple[str, float]:
        # If patch provided, inject at the specified layer(s) during forward.
        patch_store.clear()
        if patch:
            for li, vec in patch.items():
                patch_store[int(li)] = torch.tensor(vec, dtype=torch.float32)

        docs = normalize_docs(inst["docs"])
        if args.max_docs is not None:
            docs = docs[:args.max_docs]
        prompt = render_prompt(args.template, inst["question"], docs, tokenizer=tokenizer)
        enc = to_device(tokenizer(prompt, return_tensors="pt"), input_dev)

        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True)
        pred, conf = route_from_hidden(router, scaler, feat_type, router_layers, out.hidden_states)

        patch_store.clear()
        del enc, out
        return pred, conf

    all_results = {}

    for src_mode, tgt_mode in PATCH_DIRECTIONS:
        srcs = instances[src_mode]
        tgts = instances[tgt_mode]
        if not srcs or not tgts:
            print(f"Skipping {src_mode}->{tgt_mode}: insufficient instances")
            continue

        n_pairs = min(args.n_pairs, len(srcs), len(tgts))
        # sample without replacement (stable randomness)
        src_sel = random.sample(srcs, n_pairs)
        tgt_sel = random.sample(tgts, n_pairs)

        records = []
        flip_to_src = 0
        changed = 0

        for i in tqdm(range(n_pairs), desc=f"{src_mode}->{tgt_mode}"):
            tgt = tgt_sel[i]
            src = src_sel[i]

            # baseline on target
            base_pred, base_conf = route_from_forward(tgt, patch=None)

            # extract source router-layer hidden(s)
            hs_src, _, _ = forward_hidden(src)
            src_patch = pack_router_state(hs_src)

            # patched run on target
            patched_pred, patched_conf = route_from_forward(tgt, patch=src_patch)

            rec = {
                "src_id": src["id"],
                "tgt_id": tgt["id"],
                "src_mode": src_mode,
                "tgt_mode": tgt_mode,
                "baseline_pred": base_pred,
                "baseline_conf": base_conf,
                "patched_pred": patched_pred,
                "patched_conf": patched_conf,
                "flipped_to_src": (patched_pred == src_mode),
                "changed": (patched_pred != base_pred),
                "router_layers": router_layers,
                "feature_type": feat_type,
            }
            records.append(rec)

            flip_to_src += int(rec["flipped_to_src"])
            changed += int(rec["changed"])

            if (i + 1) % 15 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        all_results[f"{src_mode}_to_{tgt_mode}"] = {
            "n_pairs": n_pairs,
            "flip_rate_to_src": float(flip_to_src / max(1, n_pairs)),
            "change_rate": float(changed / max(1, n_pairs)),
            "records": records,
        }
        print(f"{src_mode}->{tgt_mode}: flip_to_src={flip_to_src/n_pairs:.3f}, changed={changed/n_pairs:.3f}")

    for hk in hooks:
        hk.remove()

    payload = {
        "meta": {
            "model": args.model,
            "router_pkl": args.router_pkl,
            "split": args.split,
            "template": args.template,
            "n_pairs": args.n_pairs,
            "seed": args.seed,
            "router_layers": router_layers,
            "feature_type": feat_type,
        },
        "results": all_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Saved -> {out_path}")

if __name__ == "__main__":
    main()
