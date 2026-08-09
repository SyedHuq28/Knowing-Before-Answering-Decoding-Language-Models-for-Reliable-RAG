import argparse
import json
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Imports from prompts.py  (graceful fallback if not present)
# ---------------------------------------------------------------------------
try:
    from prompts import DECODE_CFG, REFUSE_STRING, CONFLICT_STRING
except ImportError:
    REFUSE_STRING   = "Not enough information."
    CONFLICT_STRING = "Documents contain conflicting information."
    DECODE_CFG      = {"max_new_tokens": 256, "do_sample": False, "temperature": 1.0}

# TRIAGEINSTR never in prompts.py — always local
TRIAGEINSTR = (
    "You must use ONLY the provided documents. Do NOT use outside knowledge.\n"
    f"If the documents lack sufficient information, respond exactly: {REFUSE_STRING}\n"
    f"If the documents contain conflicting information, respond exactly: {CONFLICT_STRING}\n"
    "Otherwise, answer directly using only the documents.\n"
    "If documents conflict, do NOT choose a side and do NOT guess."
)

# ---------------------------------------------------------------------------
# FIX 1: Integer label mapping
# Your dataset uses 0=answer, 1=refuse, 2=conflict as integers.
# We normalise everything to string labels immediately on load.
# ---------------------------------------------------------------------------
INT_TO_LABEL: Dict[Any, str] = {
    0: "answer",  "0": "answer",
    1: "refuse",  "1": "refuse",
    2: "conflict","2": "conflict",
    "answer":   "answer",
    "refuse":   "refuse",
    "conflict": "conflict",
}

def normalise_label(raw) -> str:
    """Convert int or string label to canonical string."""
    if raw in INT_TO_LABEL:
        return INT_TO_LABEL[raw]
    # fallback: treat unknown as answer
    return "answer"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, str] = {
    "qwen3-4b":   "Qwen/Qwen3-4B",
    "qwen3-8b":   "Qwen/Qwen3-8B",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.2",
    "granite-8b": "ibm-granite/granite-3.1-8b-instruct",
    "selfrag":    "selfrag/selfrag_llama2_7b",
    "chatqa":     "nvidia/Llama3-ChatQA-1.5-8B",
    "o5":         "allenai/OLMo-3-7B-Instruct"
}


def get_model_id(model_arg: str) -> str:
    return MODEL_REGISTRY.get(model_arg, model_arg)


# ---------------------------------------------------------------------------
# Tokenizer / model loading  (local path → HF hub fallback)
# ---------------------------------------------------------------------------

def load_tokenizer(model_id: str, local_model_path: Optional[str] = None):
    if local_model_path and Path(local_model_path).exists():
        try:
            tok = AutoTokenizer.from_pretrained(
                local_model_path, trust_remote_code=True, local_files_only=True
            )
            print(f"[tokenizer] local: {local_model_path}")
            return tok
        except Exception as e:
            print(f"[tokenizer] local failed ({e}), falling back to HF hub ...")
    tok = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True, local_files_only=False
    )
    print(f"[tokenizer] hub: {model_id}")
    return tok


def load_model(model_id: str, local_model_path: Optional[str] = None):
    load_path = (
        local_model_path
        if (local_model_path and Path(local_model_path).exists())
        else model_id
    )
    print(f"[model] loading: {load_path}")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _truncate(text: str, tokenizer, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) > max_tokens:
        ids = ids[:max_tokens]
        text = tokenizer.decode(ids, skip_special_tokens=True)
    return text


def _apply_chat_template(content: str, tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"<|user|>\n{content}\n<|assistant|>\n"


def build_ctx_prompt(
    question: str,
    docs: List[str],
    tokenizer,
    max_ctx_tokens: int = 3500,
) -> str:
    """Standard RAG prompt: triage instruction + documents + question."""
    doc_block = "\n\n".join(f"[Document {i+1}]\n{d}" for i, d in enumerate(docs))
    content = (
        f"{TRIAGEINSTR}\n\n"
        f"Documents:\n{doc_block}\n\n"
        f"Question: {question}"
    )
    return _apply_chat_template(_truncate(content, tokenizer, max_ctx_tokens), tokenizer)


def build_noctx_prompt(
    question: str,
    tokenizer,
    max_ctx_tokens: int = 3500,
) -> str:
    """
    No-context prompt: question ONLY, no documents.
    Group 1 / 'origin without context' — pure parametric memory.
    """
    content = (
        "Answer the following question to the best of your knowledge.\n\n"
        f"Question: {question}"
    )
    return _apply_chat_template(_truncate(content, tokenizer, max_ctx_tokens), tokenizer)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _normalise_text(text: str) -> str:
    """Lowercase + strip punctuation for soft matching."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def token_f1(pred: str, gold: str) -> float:
    """Soft token-level F1 between prediction and gold answer string."""
    pred_toks = _normalise_text(pred).split()
    gold_toks = _normalise_text(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    p = n / len(pred_toks)
    r = n / len(gold_toks)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> int:
    return int(_normalise_text(pred) == _normalise_text(gold))


def classify_label(text: str) -> str:
    """
    Hard string search for refuse/conflict signals.
    Priority: refuse > conflict > answer (default).
    """
    t = text.lower()
    if (REFUSE_STRING.lower() in t
            or "not enough information" in t
            or "insufficient information" in t
            or "cannot answer" in t
            or "no information" in t):
        return "refuse"
    if (CONFLICT_STRING.lower() in t
            or "conflicting" in t
            or "contradict" in t
            or "inconsistent" in t):
        return "conflict"
    return "answer"


def score_output(
    pred_text: str,
    gold_label: str,          # already normalised string: "answer"/"refuse"/"conflict"
    gold_answer: Optional[str],
) -> Dict[str, Any]:
    """
    Compute all scores for one (pred, gold) pair.

    For refuse/conflict gold labels:
        correct = pred_label matches gold_label (hard string search)
        f1 / em = 0  (not applicable)

    For answer gold labels:
        correct      = pred_label is "answer"
        soft_correct = pred_label is "answer" AND token F1 >= 0.3
        f1 / em      = computed against gold_answer string
    """
    pred_label    = classify_label(pred_text)
    label_correct = int(pred_label == gold_label)

    f1, em = 0.0, 0
    if gold_label == "answer" and gold_answer:
        f1 = token_f1(pred_text, gold_answer)
        em = exact_match(pred_text, gold_answer)
        # soft_correct: predicted "answer" AND reasonable overlap with gold
        soft_correct = int(pred_label == "answer" and f1 >= 0.3)
    else:
        soft_correct = label_correct

    return {
        "pred_label":    pred_label,
        "label_correct": label_correct,
        "soft_correct":  soft_correct,
        "token_f1":      round(f1, 4),
        "exact_match":   em,
    }


# ---------------------------------------------------------------------------
# CAD decoding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_cad(
    ctx_prompt: str,
    noctx_prompt: str,
    model,
    tokenizer,
    alpha: float = 0.5,
    max_new_tokens: int = 256,
    temperature: float = 1.0,
) -> Tuple[str, str, str]:
    """
    Returns (cad_text, ctx_text, noctx_text).

    cad_text   — token-by-token CAD: logits(ctx) - alpha * logits(noctx)
    ctx_text   — greedy with documents  (standard RAG)
    noctx_text — greedy without documents  (Group 1 / no-context)
    """
    device = next(model.parameters()).device
    eos_id = tokenizer.eos_token_id

    ctx_enc   = tokenizer(ctx_prompt,   return_tensors="pt").to(device)
    noctx_enc = tokenizer(noctx_prompt, return_tensors="pt").to(device)

    ctx_in_len   = ctx_enc["input_ids"].shape[1]
    noctx_in_len = noctx_enc["input_ids"].shape[1]

    # --- Greedy CTX (standard RAG baseline) ---
    ctx_out = model.generate(
        **ctx_enc, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=eos_id,
    )
    ctx_text = tokenizer.decode(
        ctx_out[0][ctx_in_len:], skip_special_tokens=True
    ).strip()

    # --- Greedy NOCTX (Group 1 / origin without context) ---
    noctx_out = model.generate(
        **noctx_enc, max_new_tokens=max_new_tokens,
        do_sample=False, pad_token_id=eos_id,
    )
    noctx_text = tokenizer.decode(
        noctx_out[0][noctx_in_len:], skip_special_tokens=True
    ).strip()

    # --- CAD: token-by-token decoding ---
    ctx_run   = ctx_enc["input_ids"].clone()
    noctx_run = noctx_enc["input_ids"].clone()
    cad_ids: List[int] = []

    for _ in range(max_new_tokens):
        logits_ctx   = model(input_ids=ctx_run).logits[:, -1, :]
        logits_noctx = model(input_ids=noctx_run).logits[:, -1, :]
        logits_cad   = logits_ctx - alpha * logits_noctx
        if temperature != 1.0:
            logits_cad = logits_cad / temperature
        next_tok = logits_cad.argmax(dim=-1, keepdim=True)
        tok_id   = next_tok.item()
        cad_ids.append(tok_id)
        ctx_run   = torch.cat([ctx_run,   next_tok], dim=1)
        noctx_run = torch.cat([noctx_run, next_tok], dim=1)
        if tok_id == eos_id:
            break

    cad_text = tokenizer.decode(cad_ids, skip_special_tokens=True).strip()
    return cad_text, ctx_text, noctx_text


# ---------------------------------------------------------------------------
# Data loading  (FIX 1 applied here: label normalised immediately)
# ---------------------------------------------------------------------------

def load_instances(path: str) -> List[Dict[str, Any]]:
    instances = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            inst = json.loads(line)
            # Normalise label to string immediately — handles int 0/1/2
            raw_label = inst.get("label", inst.get("mode", "answer"))
            inst["label"] = normalise_label(raw_label)
            instances.append(inst)
    return instances


# ---------------------------------------------------------------------------
# Smoke-test verbose printer
# ---------------------------------------------------------------------------

SEP = "─" * 72

def print_sample(
    idx: int,
    inst: Dict,
    scores: Dict[str, Dict],
    outputs: Dict[str, str],
):
    print(f"\n{SEP}")
    print(
        f"SAMPLE {idx+1}  "
        f"|  gold_label={inst['label']}  "
        f"|  gold_answer={inst.get('gold_answer', 'N/A')}"
    )
    print(f"Question: {inst['question'][:120]}")
    print(SEP)
    for key in ["noctx", "ctx", "cad"]:
        tag = {
            "noctx": "GROUP-1 (no context / parametric memory)",
            "ctx":   "CTX     (standard RAG greedy)           ",
            "cad":   "CAD     (context-aware decoding)        ",
        }[key]
        s   = scores[key]
        out = outputs[key]
        print(f"\n  [{tag}]")
        print(
            f"  pred_label   : {s['pred_label']}\n"
            f"  label_correct: {s['label_correct']}  |  "
            f"soft_correct: {s['soft_correct']}  |  "
            f"token_F1: {s['token_f1']:.3f}  |  "
            f"EM: {s['exact_match']}"
        )
        print(f"  output       : {out[:300]}")
    print(SEP)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    instances: List[Dict[str, Any]],
    model,
    tokenizer,
    args,
) -> List[Dict[str, Any]]:

    results = []
    n       = len(instances)
    verbose = args.smoke_test

    for idx, inst in enumerate(instances):
        question    = inst["question"]
        docs        = inst.get("docs", [])
        gold_label  = inst["label"]          # already normalised string
        gold_answer = inst.get("gold_answer", inst.get("answer", None))

        ctx_prompt   = build_ctx_prompt(question, docs, tokenizer, args.max_ctx_tokens)
        noctx_prompt = build_noctx_prompt(question, tokenizer, args.max_ctx_tokens)

        cad_text, ctx_text, noctx_text = generate_cad(
            ctx_prompt     = ctx_prompt,
            noctx_prompt   = noctx_prompt,
            model          = model,
            tokenizer      = tokenizer,
            alpha          = args.alpha,
            max_new_tokens = DECODE_CFG.get("max_new_tokens", 256),
            temperature    = DECODE_CFG.get("temperature", 1.0),
        )

        scores = {
            "cad":   score_output(cad_text,   gold_label, gold_answer),
            "ctx":   score_output(ctx_text,   gold_label, gold_answer),
            "noctx": score_output(noctx_text, gold_label, gold_answer),
        }

        if verbose:
            print_sample(
                idx, inst, scores,
                {"cad": cad_text, "ctx": ctx_text, "noctx": noctx_text},
            )

        results.append({
            "id":          inst.get("id", idx),
            "question":    question,
            "gold_label":  gold_label,
            "gold_answer": gold_answer,
            # CAD
            "cad_output":        cad_text,
            "cad_pred_label":    scores["cad"]["pred_label"],
            "cad_label_correct": scores["cad"]["label_correct"],
            "cad_soft_correct":  scores["cad"]["soft_correct"],
            "cad_token_f1":      scores["cad"]["token_f1"],
            "cad_em":            scores["cad"]["exact_match"],
            # CTX (standard RAG)
            "ctx_output":        ctx_text,
            "ctx_pred_label":    scores["ctx"]["pred_label"],
            "ctx_label_correct": scores["ctx"]["label_correct"],
            "ctx_soft_correct":  scores["ctx"]["soft_correct"],
            "ctx_token_f1":      scores["ctx"]["token_f1"],
            "ctx_em":            scores["ctx"]["exact_match"],
            # NOCTX (Group 1)
            "noctx_output":        noctx_text,
            "noctx_pred_label":    scores["noctx"]["pred_label"],
            "noctx_label_correct": scores["noctx"]["label_correct"],
            "noctx_soft_correct":  scores["noctx"]["soft_correct"],
            "noctx_token_f1":      scores["noctx"]["token_f1"],
            "noctx_em":            scores["noctx"]["exact_match"],
        })

        if not verbose and (idx + 1) % 25 == 0:
            def _acc(key):
                return sum(r[f"{key}_label_correct"] for r in results) / len(results)
            print(
                f"  [{idx+1}/{n}] label_acc — "
                f"CAD={_acc('cad'):.3f}  "
                f"CTX={_acc('ctx'):.3f}  "
                f"NOCTX={_acc('noctx'):.3f}"
            )

    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    n = len(results)

    for prefix in ["cad", "ctx", "noctx"]:
        metrics[f"{prefix}_label_acc"]     = sum(r[f"{prefix}_label_correct"] for r in results) / n
        metrics[f"{prefix}_soft_acc"]      = sum(r[f"{prefix}_soft_correct"]  for r in results) / n
        metrics[f"{prefix}_mean_token_f1"] = sum(r[f"{prefix}_token_f1"]      for r in results) / n
        metrics[f"{prefix}_mean_em"]       = sum(r[f"{prefix}_em"]            for r in results) / n

        # Per-class label F1 / precision / recall
        by_class: Dict[str, Dict[str, int]] = {}
        for r in results:
            g = r["gold_label"]
            p = r[f"{prefix}_pred_label"]
            for cls in set([g, p]):
                if cls not in by_class:
                    by_class[cls] = {"tp": 0, "fp": 0, "fn": 0}
            if g == p:
                by_class[g]["tp"] += 1
            else:
                by_class[g]["fn"] += 1
                by_class[p]["fp"] += 1

        for cls, c in by_class.items():
            tp, fp, fn = c["tp"], c["fp"], c["fn"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            metrics[f"{prefix}_{cls}_label_f1"]        = round(f1,   4)
            metrics[f"{prefix}_{cls}_label_precision"] = round(prec, 4)
            metrics[f"{prefix}_{cls}_label_recall"]    = round(rec,  4)

        # Token-F1 broken down by gold label
        for cls in ["answer", "refuse", "conflict"]:
            subset = [r for r in results if r["gold_label"] == cls]
            if subset:
                metrics[f"{prefix}_{cls}_token_f1"] = round(
                    sum(r[f"{prefix}_token_f1"] for r in subset) / len(subset), 4
                )

    return metrics


def print_metrics(metrics: Dict[str, float]):
    print("\n" + "=" * 72)
    print("METRICS")
    print("=" * 72)
    for prefix, label in [
        ("cad",   "CAD   (context-aware decoding)"),
        ("ctx",   "CTX   (standard RAG greedy)   "),
        ("noctx", "NOCTX (no context / Group 1)  "),
    ]:
        print(f"\n  [{label}]")
        for k, v in sorted(metrics.items()):
            if k.startswith(prefix + "_"):
                print(f"    {k[len(prefix)+1:]:40s}: {v:.4f}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Save — always, regardless of smoke_test flag
# ---------------------------------------------------------------------------

def save_results(
    results: List[Dict[str, Any]],
    metrics: Dict[str, float],
    output_path: str,
):
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    metrics_path = out_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults  → {out_path}")
    print(f"Metrics  → {metrics_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="CAD RAG inference + evaluation")
    p.add_argument("--model",          required=True,
                   help="Registry key (e.g. qwen3-4b) or raw HF hub ID")
    p.add_argument("--model_path",     default=None,
                   help="Optional local weights path; falls back to HF hub on failure")
    p.add_argument("--data",           required=True,
                   help="Path to instances JSONL file")
    p.add_argument("--output",         default="results_cad.jsonl",
                   help="Output JSONL path (always written)")
    p.add_argument("--alpha",          type=float, default=0.5,
                   help="CAD subtraction weight α  (0=vanilla RAG, 1=full subtraction)")
    p.add_argument("--max_ctx_tokens", type=int,   default=3500,
                   help="Max tokens in the context block")
    p.add_argument("--limit",          type=int,   default=None,
                   help="Evaluate only first N instances (full run)")
    p.add_argument("--smoke_test",     action="store_true",
                   help="Run first --smoke_n samples with verbose per-sample output")
    p.add_argument("--smoke_n",        type=int,   default=5,
                   help="Number of samples in smoke test (default: 5)")
    return p.parse_args()


def main():
    args     = parse_args()
    model_id = get_model_id(args.model)

    print(f"\nModel   : {model_id}")
    print(f"Data    : {args.data}")
    print(f"Alpha   : {args.alpha}")
    print(f"Output  : {args.output}  (always saved)")
    print(
        f"Mode    : "
        f"{'SMOKE TEST (' + str(args.smoke_n) + ' samples)' if args.smoke_test else 'FULL RUN'}\n"
    )

    tokenizer = load_tokenizer(model_id, local_model_path=args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(model_id, local_model_path=args.model_path)

    instances = load_instances(args.data)   # labels already normalised here

    if args.smoke_test:
        instances = instances[: args.smoke_n]
    elif args.limit:
        instances = instances[: args.limit]

    print(f"Instances to evaluate: {len(instances)}\n")

    results = evaluate(instances, model, tokenizer, args)
    metrics = compute_all_metrics(results)

    print_metrics(metrics)
    save_results(results, metrics, args.output)


if __name__ == "__main__":
    main()
