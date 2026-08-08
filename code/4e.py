import argparse
import gzip
import json
import logging
import random
import re
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import render_prompt, DECODE_CFG, REFUSE_STRING, CONFLICT_STRING


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def make_logger(log_file: Optional[str]) -> logging.Logger:
    logger = logging.getLogger("steering")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ──────────────────────────────────────────────────────────────────────────────
# Regex mode classifier (NOT exact string match; uses regex patterns)
# ──────────────────────────────────────────────────────────────────────────────
MODE_PATTERNS = {
    "refuse": re.compile(
        r"\b("
        r"not enough (information|evidence)|insufficient (information|evidence)|"
        r"cannot (answer|determine|verify)|can't (answer|determine|verify)|"
        r"don't (know|have enough)|no (information|evidence)|"
        r"unable to (answer|determine|verify)|"
        r"cannot be answered from the provided (documents|context)|"
        r"provided documents do not (contain|include) (enough|sufficient)"
        r")\b",
        re.IGNORECASE,
    ),
    "conflict": re.compile(
        r"\b("
        r"conflict|contradict|contradiction|inconsistent|disagree|"
        r"mutually incompatible|cannot reconcile|at odds|"
        r"two (sources|documents) (disagree|contradict)|"
        r"conflicting (information|claims)"
        r")\b",
        re.IGNORECASE,
    ),
}

def classify_mode_regex(text: str) -> str:
    t = text.strip()
    if MODE_PATTERNS["conflict"].search(t):
        return "conflict"
    if MODE_PATTERNS["refuse"].search(t):
        return "refuse"
    return "answer"


# ──────────────────────────────────────────────────────────────────────────────
# Model-agnostic helpers
# ──────────────────────────────────────────────────────────────────────────────
def get_layers(m):
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers  # Llama/Mistral/Qwen/Phi style
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h  # GPT-2 style
    if hasattr(m, "model") and hasattr(m.model, "decoder") and hasattr(m.model.decoder, "layers"):
        return m.model.decoder.layers  # OPT style
    raise AttributeError(f"Cannot find decoder layers for {type(m)}")

def get_mlp(block):
    if hasattr(block, "mlp"):
        return block.mlp
    if hasattr(block, "ffn"):
        return block.ffn
    if hasattr(block, "feed_forward"):
        return block.feed_forward
    raise AttributeError(f"Cannot find MLP module in block type {type(block)}")

def get_input_device(model):
    # device_map="auto" safe input placement
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        for k in ["model.embed_tokens", "model.model.embed_tokens", "transformer.wte"]:
            if k in model.hf_device_map:
                return torch.device(model.hf_device_map[k])
        return torch.device(next(iter(model.hf_device_map.values())))
    return next(model.parameters()).device


# ──────────────────────────────────────────────────────────────────────────────
# Probe-driven layer selection
# ──────────────────────────────────────────────────────────────────────────────
def load_probe_scores(results_dir: Path, steer_space: str, metric: str) -> Dict[int, float]:
    fname = "layer_probe_hidden.json" if steer_space == "hidden" else "layer_probe_mlp.json"
    path = results_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing probe file: {path}. Run 3_layer_probe_sweep.py first.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scores: Dict[int, float] = {}
    for k, v in data.items():
        layer = int(k)
        if metric not in v:
            raise KeyError(f"Metric '{metric}' missing for layer {layer} in {path}. Keys={list(v.keys())}")
        scores[layer] = float(v[metric])
    return scores

def parse_layers_arg(
    layers_arg_list: List[str],
    L: int,
    results_dir: Path,
    steer_space: str,
    metric: str,
    include_neighbors: bool,
    logger: logging.Logger,
) -> List[int]:
    # --layers all
    if len(layers_arg_list) == 1 and layers_arg_list[0].lower() == "all":
        layers = list(range(L))
        logger.info(f"Layer selection: all ({len(layers)} layers)")
        return layers

    # --layers top 10
    if len(layers_arg_list) >= 2 and layers_arg_list[0].lower() in {"top", "topk"}:
        k = int(layers_arg_list[1])
        scores = load_probe_scores(results_dir, steer_space, metric)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        layers = [l for l, _ in ranked[:k]]
        logger.info(f"Layer selection: top {k} by {metric} → {layers}")

    # --layers top10
    elif len(layers_arg_list) == 1 and layers_arg_list[0].lower().startswith("top"):
        k = int(layers_arg_list[0].lower().replace("top", "").strip())
        scores = load_probe_scores(results_dir, steer_space, metric)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        layers = [l for l, _ in ranked[:k]]
        logger.info(f"Layer selection: top{k} by {metric} → {layers}")

    # --layers 18 19 20
    else:
        layers = [int(x) for x in layers_arg_list]
        logger.info(f"Layer selection: explicit → {layers}")

    layers = sorted(set([l for l in layers if 0 <= l < L]))

    if include_neighbors:
        expanded = set(layers)
        for l in layers:
            if l - 1 >= 0:
                expanded.add(l - 1)
            if l + 1 < L:
                expanded.add(l + 1)
        layers = sorted(expanded)
        logger.info(f"Include neighbors ±1 → {len(layers)} layers: {layers}")

    return layers


# ──────────────────────────────────────────────────────────────────────────────
# RMS normalization (NumPy 2.0 safe; chunked)
# ──────────────────────────────────────────────────────────────────────────────
def load_or_compute_rms(
    activations_dir: Path,
    steering_cache_dir: Path,
    space_key: str,
    rms_n: int,
    L: int,
    logger: logging.Logger,
) -> np.ndarray:
    """
    rms[layer] = mean over examples of sqrt(mean(x^2 over dims)) at that layer.
    Computed from activations/{H|M}_P0_train.npy (memmapped), in chunks.
    """
    rms_path = steering_cache_dir / f"rms_by_layer_{space_key}.npy"
    if rms_path.exists():
        rms = np.load(rms_path).astype(np.float32)
        logger.info(f"Loaded RMS cache: {rms_path}")
        return rms[:L]

    act_file = activations_dir / (f"H_P0_train.npy" if space_key == "hidden" else f"M_P0_train.npy")
    if not act_file.exists():
        raise FileNotFoundError(
            f"Need {act_file} to estimate RMS. Run 2_extract_allayer.py for P0/train first."
        )

    X = np.load(act_file, mmap_mode="r")  # [N, Lfile, D]
    n = min(rms_n, X.shape[0])
    Lfile = X.shape[1]
    Luse = min(L, Lfile)

    logger.info(f"Computing RMS from {act_file} with n={n} (chunked), Luse={Luse}")

    bs = 64
    sum_rms = np.zeros((Luse,), dtype=np.float64)
    count = 0

    for start in range(0, n, bs):
        end = min(start + bs, n)
        chunk = np.asarray(X[start:end, :Luse, :], dtype=np.float32)
        rms_chunk = np.sqrt(np.mean(chunk * chunk, axis=2))  # [B, Luse]
        sum_rms += rms_chunk.sum(axis=0)
        count += rms_chunk.shape[0]

    rms = (sum_rms / max(1, count)).astype(np.float32)
    np.save(rms_path, rms)
    logger.info(f"Saved RMS → {rms_path}")
    return rms[:Luse]


# ──────────────────────────────────────────────────────────────────────────────
# Docs normalization (defensive)
# ──────────────────────────────────────────────────────────────────────────────
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
# UNIFIED TRIAGE INSTRUCTION (Option A)
# ──────────────────────────────────────────────────────────────────────────────
def build_unified_triage_instruction() -> str:
    return (
        "You must use ONLY the provided documents. Do NOT use outside knowledge.\n"
        f"If the documents lack sufficient information, respond exactly: {REFUSE_STRING}\n"
        f"If the documents contain conflicting information, respond exactly: {CONFLICT_STRING}\n"
        "Otherwise, answer directly using only the documents.\n"
        "If documents conflict, do NOT choose a side and do NOT guess."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--template", default="P0")

    parser.add_argument("--steer_mode", default="refuse", choices=["refuse", "conflict"])
    parser.add_argument("--steer_space", default="hidden", choices=["hidden", "mlp"])

    parser.add_argument(
        "--layers", nargs="+", default=["top", "10"],
        help='Examples: --layers all | --layers 18 19 20 | --layers top 10 | --layers top10'
    )
    parser.add_argument("--include_neighbors", type=int, default=0)

    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--last_token_only", type=int, default=1)
    parser.add_argument("--prompt_step_only", type=int, default=0)

    parser.add_argument("--alpha_unit", default="rms", choices=["rms", "none"])
    parser.add_argument("--rms_n", type=int, default=2000)

    parser.add_argument("--model", default="/users/adhp263/sharedscratch/models/Ministral-3-3B-Base-2512")

    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--top_metric", default="val_acc", choices=["val_acc", "val_f1"])

    parser.add_argument("--save_dir", default="steering")
    parser.add_argument("--save_gens", type=int, default=1)
    parser.add_argument("--log_file", default=None)

    # NEW: unified triage prompt switch
    parser.add_argument("--use_unified_triage", type=int, default=1,
                        help="1 = always include unified triage instruction (Option A).")
    args = parser.parse_args()

    logger = make_logger(args.log_file)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    steering_out_dir = Path(args.save_dir)
    steering_out_dir.mkdir(exist_ok=True)

    steering_cache_dir = Path("steering")  # vectors + rms cache live here
    activations_dir = Path("activations")
    results_dir = Path(args.results_dir)

    logger.info(f"Run config: model={args.model} template={args.template} split={args.split}")
    logger.info(f"Steering: mode={args.steer_mode} space={args.steer_space} alpha_unit={args.alpha_unit}")
    logger.info(f"Flags: last_token_only={bool(args.last_token_only)} prompt_step_only={bool(args.prompt_step_only)}")
    logger.info(f"Unified triage prompt: {bool(args.use_unified_triage)}")
    logger.info("Eval: regex expressions (not exact string match).")

    # Load steering vectors
    vec_file = "steering_vectors_hidden.npy" if args.steer_space == "hidden" else "steering_vectors_mlp.npy"
    vec_path = steering_cache_dir / vec_file
    if not vec_path.exists():
        raise FileNotFoundError(f"Missing steering vectors: {vec_path}. Run 4_compute_steering_vectors.py first.")
    V_all = np.load(vec_path).astype(np.float32)  # [Lvec,2,D]

    MODE_IDX = {"refuse": 0, "conflict": 1}
    m_idx = MODE_IDX[args.steer_mode]

    # Load model
    logger.info("Loading model weights...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    blocks = get_layers(model)
    L = len(blocks)
    D_model = model.config.hidden_size
    input_device = get_input_device(model)

    # Fail early if vectors mismatch model
    D_vec = V_all.shape[2]
    L_vec = V_all.shape[0]
    logger.info(f"Model: L={L} D={D_model} | Vectors: L={L_vec} D={D_vec} | input_device={input_device}")
    if D_vec != D_model:
        raise RuntimeError(
            f"Vector dim mismatch: vectors D={D_vec}, model hidden_size D={D_model}.\n"
            f"Fix: recompute activations + steering vectors for THIS model:\n"
            f"  python 2_extract_allayer.py --model {args.model} --template P0 --split train\n"
            f"  python 4_compute_steering_vectors.py"
        )

    # Choose layers
    sweep_layers = parse_layers_arg(
        args.layers, L=L, results_dir=results_dir, steer_space=args.steer_space,
        metric=args.top_metric, include_neighbors=bool(args.include_neighbors), logger=logger
    )

    # RMS normalization
    rms_by_layer = None
    if args.alpha_unit == "rms":
        rms_by_layer = load_or_compute_rms(
            activations_dir=activations_dir,
            steering_cache_dir=steering_cache_dir,
            space_key=args.steer_space,
            rms_n=args.rms_n,
            L=L,
            logger=logger,
        )

    # Load instances (only true_mode == steer_mode)
    inst_path = Path(f"instances_{args.split}.jsonl")
    if not inst_path.exists():
        raise FileNotFoundError(f"Missing {inst_path}. Run 0_build_instances.py first.")

    instances = []
    with open(inst_path, "r", encoding="utf-8") as f:
        for line in f:
            inst = json.loads(line)
            if inst.get("true_mode") == args.steer_mode:
                instances.append(inst)

    random.shuffle(instances)
    if args.n > 0:
        instances = instances[:min(args.n, len(instances))]

    logger.info(f"Loaded {len(instances)} instances where true_mode={args.steer_mode} from {inst_path}")
    if len(instances) == 0:
        raise RuntimeError("No instances found for this mode/split. Check your instances_*.jsonl.")

    # Unified instruction (same for all modes)
    instruction = build_unified_triage_instruction() if args.use_unified_triage else ""

    # Steering state shared across hooks
    current_steer = {
        "layer": None,
        "vector": None,       # torch [1,1,D] float32
        "alpha_eff": 0.0,     # scalar actually applied
        "last_token_only": bool(args.last_token_only),
        "prompt_step_only": bool(args.prompt_step_only),
        "prompt_len": 0,
    }

    def make_steer_hook(layer_idx: int):
        def hook(_module, _input, output):
            if current_steer["layer"] != layer_idx:
                return None
            if current_steer["alpha_eff"] == 0.0:
                return None

            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output  # [B,T,D]
            if not torch.is_tensor(h) or h.ndim != 3:
                return None

            # prefill-only steering (mechanistic)
            if current_steer["prompt_step_only"]:
                if h.shape[1] != current_steer["prompt_len"]:
                    return None

            v = current_steer["vector"].to(h.device).to(h.dtype)[0, 0, :]
            a = current_steer["alpha_eff"]

            h2 = h.clone()
            if current_steer["last_token_only"]:
                h2[:, -1, :] = h2[:, -1, :] + a * v
            else:
                h2 = h2 + a * v.view(1, 1, -1)

            if is_tuple:
                return (h2,) + output[1:]
            return h2
        return hook

    # Register hooks for all layers (only active when current_steer["layer"] matches)
    hooks = []
    for li, block in enumerate(blocks):
        if args.steer_space == "hidden":
            hooks.append(block.register_forward_hook(make_steer_hook(li)))
        else:
            hooks.append(get_mlp(block).register_forward_hook(make_steer_hook(li)))
    logger.info(f"Registered {len(hooks)} hooks.")

    # Output tags / files
    layer_tag = "all" if len(sweep_layers) == L else f"{len(sweep_layers)}layers"
    alpha_tag = f"{len(args.alphas)}a"
    triage_tag = f"triage{int(args.use_unified_triage)}"
    tag = (f"{args.template}_{args.split}_{args.steer_space}_{args.steer_mode}"
           f"_{triage_tag}_unit{args.alpha_unit}"
           f"_lto{int(args.last_token_only)}_pso{int(args.prompt_step_only)}"
           f"_{layer_tag}_{alpha_tag}_n{len(instances)}")

    summary_path = steering_out_dir / f"steering_sweep_{tag}.json"
    gens_path = steering_out_dir / f"steering_gens_{tag}.jsonl.gz"

    gens_f = gzip.open(gens_path, "wt", encoding="utf-8") if args.save_gens else None
    if gens_f:
        logger.info(f"Saving generations → {gens_path}")
    logger.info(f"Saving summary → {summary_path}")

    logger.info(f"Sweep layers={len(sweep_layers)} alphas={args.alphas} n={len(instances)}")

    results = []
    pbar = tqdm(total=len(sweep_layers) * len(args.alphas), desc="sweep")

    # Sweep
    for layer in sweep_layers:
        if layer < 0 or layer >= L or layer >= V_all.shape[0]:
            logger.info(f"Skipping L{layer} (out of range for model/vectors)")
            pbar.update(len(args.alphas))
            continue

        v_tensor = torch.tensor(V_all[layer, m_idx], dtype=torch.float32).view(1, 1, -1)

        for alpha in args.alphas:
            counts = {"answer": 0, "refuse": 0, "conflict": 0}
            hits = 0
            false_answers = 0

            for inst in instances:
                docs = normalize_docs(inst.get("docs", []))
                prompt = render_prompt(
                    args.template,
                    inst["question"],
                    docs,
                    instruction=instruction,
                    tokenizer=tokenizer
                )

                enc = tokenizer(prompt, return_tensors="pt")
                enc = {k: v.to(input_device) for k, v in enc.items()}
                inp_len = enc["input_ids"].shape[1]

                # alpha normalization
                alpha_eff = float(alpha)
                if args.alpha_unit == "rms":
                    alpha_eff = float(alpha) * float(rms_by_layer[layer])

                current_steer.update({
                    "layer": layer,
                    "vector": v_tensor,
                    "alpha_eff": alpha_eff,
                    "prompt_len": int(inp_len),
                })

                with torch.inference_mode():
                    out = model.generate(
                        **enc,
                        max_new_tokens=DECODE_CFG["max_new_tokens"],
                        do_sample=DECODE_CFG["do_sample"],
                        temperature=DECODE_CFG["temperature"],
                        top_p=DECODE_CFG["top_p"],
                        repetition_penalty=DECODE_CFG["repetition_penalty"],
                        pad_token_id=tokenizer.eos_token_id,
                    )

                # stop steering immediately
                current_steer["alpha_eff"] = 0.0

                gen = tokenizer.decode(out[0][inp_len:], skip_special_tokens=True).strip()
                pred_mode = classify_mode_regex(gen)

                counts[pred_mode] += 1
                hits += int(pred_mode == args.steer_mode)
                false_answers += int(pred_mode == "answer")

                if gens_f:
                    gens_f.write(json.dumps({
                        "id": inst.get("id"),
                        "true_mode": inst.get("true_mode"),
                        "layer": int(layer),
                        "alpha": float(alpha),
                        "alpha_unit": args.alpha_unit,
                        "alpha_eff_scale": float(rms_by_layer[layer]) if args.alpha_unit == "rms" else 1.0,
                        "steer_space": args.steer_space,
                        "steer_mode": args.steer_mode,
                        "use_unified_triage": bool(args.use_unified_triage),
                        "last_token_only": bool(args.last_token_only),
                        "prompt_step_only": bool(args.prompt_step_only),
                        "pred_mode_regex": pred_mode,
                        "gen_text": gen,
                    }, ensure_ascii=False) + "\n")

                del enc, out

            if torch.cuda.is_available() and (layer % 8 == 0):
                torch.cuda.empty_cache()
            gc.collect()

            n = max(1, len(instances))
            compliance = hits / n
            far = false_answers / n

            results.append({
                "split": args.split,
                "template": args.template,
                "layer": int(layer),
                "alpha": float(alpha),
                "alpha_unit": args.alpha_unit,
                "alpha_eff_scale": float(rms_by_layer[layer]) if args.alpha_unit == "rms" else 1.0,
                "steer_space": args.steer_space,
                "steer_mode": args.steer_mode,
                "use_unified_triage": bool(args.use_unified_triage),
                "last_token_only": bool(args.last_token_only),
                "prompt_step_only": bool(args.prompt_step_only),
                "n": int(len(instances)),
                "compliance_regex": float(compliance),
                "false_answer_rate_regex": float(far),
                "pred_counts_regex": counts,
            })

            logger.info(
                f"L{layer:02d} α={alpha:+.2f} ({args.alpha_unit}) "
                f"→ compliance={compliance:.3f} FAR={far:.3f} counts={counts}"
            )

            pbar.update(1)

    pbar.close()

    if gens_f:
        gens_f.close()
        logger.info(f"Saved generations → {gens_path}")

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved summary → {summary_path}")

    for h in hooks:
        h.remove()

    logger.info("Done.")


if __name__ == "__main__":
    main()
