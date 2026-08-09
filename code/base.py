import argparse, json, time, gc, re
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import f1_score

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
REFUSE_STRING   = "Not enough information."
CONFLICT_STRING = "Documents contain conflicting information."
MODES           = ["answer", "refuse", "conflict"]

CHAT_PREFIXES = [
    "<|assistant|>", "<|ASSISTANT|>", "Assistant:", "ASSISTANT:",
    "<|im_start|>assistant", "[/INST]",
]

REFUSE_RE = re.compile(
    r"\b("
    r"not enough (information|evidence)|insufficient (information|evidence)|"
    r"cannot (answer|determine|verify)|can't (answer|determine|verify)|"
    r"unable to (answer|determine|verify)|"
    r"documents? do not (contain|include)|"
    r"cannot be answered from the provided"
    r")\b",
    re.IGNORECASE,
)
CONFLICT_RE = re.compile(
    r"\b(documents?|sources?|passages?|provided documents?)\b.*\b("
    r"conflict|conflicting|contradict|contradiction|inconsistent|disagree|cannot reconcile|at odds"
    r")\b",
    re.IGNORECASE,
)
CONFLICT_RE_FALLBACK = re.compile(
    r"\b(contradict|contradiction|inconsistent|conflicting information|at odds|cannot reconcile)\b",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Template — short examples, .replace() safe
# ──────────────────────────────────────────────────────────────────────────────
FAITH_COT_TEMPLATE = (
    "Task Description:\n"
    "Given a question and a context, triage the relationship between the evidence and the question.\n"
    "Respond ONLY in the JSON format shown below.\n\n"
    "You MUST reason through these steps before answering:\n"
    "1. [Self-Evidence Extraction]: List specific facts from the context that address the question.\n"
    "2. [Fact Analysis]: Check if extracted facts are consistent or explicitly contradict each other.\n"
    "3. [Context Check]: Judge if the documents provide one answer, multiple answers, or no answer.\n"
    "4. [Final Verification]: Choose the label — ANSWER, REFUSE, or CONFLICT.\n\n"
    "Triage Labels:\n"
    "- ANSWER: Context is consistent and provides one grounded answer — write the actual answer text.\n"
    '- REFUSE: Context lacks sufficient information — write exactly: "' + REFUSE_STRING + '"\n'
    '- CONFLICT: Context contains contradictions — write exactly: "' + CONFLICT_STRING + '"\n\n'
    "Response Format:\n"
    '{ "Reason": "1. [Self-Evidence Extraction]: ... 2. [Fact Analysis]: ... 3. [Context Check]: ... 4. [Final Verification]: ...", "Answer": "answer text OR exact refuse/conflict string" }\n\n'
    "---\n"
    "Example 1 (ANSWER):\n"
    "Question: What colour is the sky?\n"
    "Context: The sky appears blue due to Rayleigh scattering.\n"
    '{ "Reason": "1. [Self-Evidence Extraction]: Sky is blue (Rayleigh scattering). 2. [Fact Analysis]: One consistent fact. 3. [Context Check]: Exactly one answer. 4. [Final Verification]: Mode is ANSWER.", "Answer": "Blue" }\n\n'
    "Example 2 (REFUSE):\n"
    "Question: What is the mass of Zelonite?\n"
    "Context: Zelonite is found in volcanic regions.\n"
    '{ "Reason": "1. [Self-Evidence Extraction]: No mass mentioned. 2. [Fact Analysis]: No relevant facts. 3. [Context Check]: Context does not answer the question. 4. [Final Verification]: Mode is REFUSE.", "Answer": "' + REFUSE_STRING + '" }\n\n'
    "Example 3 (CONFLICT):\n"
    "Question: Boiling point of Zelonite?\n"
    "Context: Lab A says 340C. Lab B says 415C.\n"
    '{ "Reason": "1. [Self-Evidence Extraction]: Lab A (340C), Lab B (415C). 2. [Fact Analysis]: Different values, contradiction. 3. [Context Check]: Two contradictory answers. 4. [Final Verification]: Mode is CONFLICT.", "Answer": "' + CONFLICT_STRING + '" }\n\n'
    "---\n"
    "Your Task:\n"
    "Question: {question}\n"
    "Context: {context}\n"
    "CoT-Answer: "
)

# ──────────────────────────────────────────────────────────────────────────────
# Classifier
# ──────────────────────────────────────────────────────────────────────────────
def strip_chat_prefix(text: str) -> str:
    text = (text or "").strip()
    for prefix in CHAT_PREFIXES:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            break
    return text

def extract_json_answer(text: str):
    match = re.search(r'"Answer"\s*:\s*"(.*?)"', text, re.DOTALL)
    return match.group(1).strip() if match else None

def classify_mode(text: str) -> str:
    text = strip_chat_prefix(text)
    if not text:
        return "answer"
    json_answer = extract_json_answer(text)
    if json_answer:
        low = json_answer.lower()
        if CONFLICT_STRING.lower() in low:           return "conflict"
        if REFUSE_STRING.lower() in low:             return "refuse"
        if CONFLICT_RE.search(json_answer):          return "conflict"
        if CONFLICT_RE_FALLBACK.search(json_answer): return "conflict"
        if REFUSE_RE.search(json_answer):            return "refuse"
        return "answer"
    low = text.lower()
    if CONFLICT_STRING.lower() in low:               return "conflict"
    if REFUSE_STRING.lower() in low:                 return "refuse"
    if CONFLICT_RE.search(text):                     return "conflict"
    if CONFLICT_RE_FALLBACK.search(text):            return "conflict"
    if REFUSE_RE.search(text):                       return "refuse"
    return "answer"

# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────
def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=MODES,
                          average="macro", zero_division=0))

def acc(y_true, y_pred) -> float:
    return float(np.mean([t == p for t, p in zip(y_true, y_pred)]))

def far_refuse_conflict(y_true, y_pred) -> float:
    denom = sum(t in ("refuse", "conflict") for t in y_true)
    num   = sum((t in ("refuse", "conflict") and p == "answer")
                for t, p in zip(y_true, y_pred))
    return float(num / max(1, denom))

def answer_substring(rows, y_true) -> float:
    vals = []
    for r, t in zip(rows, y_true):
        if t != "answer":
            continue
        gold = (r.get("gold_answer") or "").strip().lower()
        raw  = r["B6"]["text"] if isinstance(r.get("B6"), dict) else (r.get("B6") or "")
        json_ans = extract_json_answer(raw)
        pred = (json_ans or strip_chat_prefix(raw)).strip().lower()
        if not gold:
            continue
        vals.append(1.0 if gold in pred else 0.0)
    return float(np.mean(vals)) if vals else 0.0

# ──────────────────────────────────────────────────────────────────────────────
# Device helper
# ──────────────────────────────────────────────────────────────────────────────
def get_input_device(m):
    if hasattr(m, "hf_device_map") and isinstance(m.hf_device_map, dict):
        for k in ["model.embed_tokens", "model.model.embed_tokens", "transformer.wte"]:
            if k in m.hf_device_map:
                return torch.device(m.hf_device_map[k])
        return torch.device(next(iter(m.hf_device_map.values())))
    return next(m.parameters()).device

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",          default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--model",          default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--instances",      default="all",
                        help="Number of instances to process (e.g. 3) or 'all'")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--max_input_tokens", type=int, default=40000,
                        help="Truncate input to this many tokens to avoid OOM")
    parser.add_argument("--out_dir",        default="results/rescored")
    args = parser.parse_args()

    # ── Model setup ──────────────────────────────────────────────────────────
    print(f"[{time.strftime('%H:%M:%S')}] Loading model {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    INPUT_DEVICE = get_input_device(model)
    print(f"[{time.strftime('%H:%M:%S')}] Model loaded. input_device={INPUT_DEVICE}")

    # ── Data loading ─────────────────────────────────────────────────────────
    all_instances = []
    with open(f"instances_{args.split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            all_instances.append(json.loads(line))

    if args.instances.lower() == "all":
        instances = all_instances
    else:
        instances = all_instances[:int(args.instances)]

    print(f"[{time.strftime('%H:%M:%S')}] "
          f"Processing {len(instances)} / {len(all_instances)} instances ({args.split})")

    # ── Inference ────────────────────────────────────────────────────────────
    records  = []
    skipped  = []
    oom_count = 0

    for idx, inst in enumerate(tqdm(instances, desc=f"B6-CoT/{args.split}")):
        q            = inst["question"]
        context_text = "\n".join(
            f"[Document {i+1}] {d.strip()}"
            for i, d in enumerate(inst["docs"])
        )
        prompt = FAITH_COT_TEMPLATE.replace("{question}", q).replace("{context}", context_text)

        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_tokens,
            )
            inputs = {k: v.to(INPUT_DEVICE) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )

            gen_text = tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip()

            # Close JSON if generation was cut off
            if gen_text.count("{") > gen_text.count("}"):
                gen_text += "}"

            records.append({
                "id":          inst["id"],
                "true_mode":   inst["true_mode"],
                "gold_answer": inst.get("gold_answer", ""),
                "B6":          {"text": gen_text},
                "skipped":     False,
            })

        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            tqdm.write(f"[OOM] Skipped {inst['id']} (OOM #{oom_count})")
            torch.cuda.empty_cache()
            gc.collect()

            # Record as skipped — classified as "answer" by default
            # so it doesn't silently bias metrics
            records.append({
                "id":          inst["id"],
                "true_mode":   inst["true_mode"],
                "gold_answer": inst.get("gold_answer", ""),
                "B6":          {"text": ""},
                "skipped":     True,
            })
            skipped.append(inst["id"])
            continue

        except Exception as e:
            tqdm.write(f"[ERROR] Skipped {inst['id']}: {e}")
            records.append({
                "id":          inst["id"],
                "true_mode":   inst["true_mode"],
                "gold_answer": inst.get("gold_answer", ""),
                "B6":          {"text": ""},
                "skipped":     True,
            })
            skipped.append(inst["id"])
            continue

        if (idx + 1) % 5 == 0:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ── Evaluation (exclude skipped) ─────────────────────────────────────────
    valid_records = [r for r in records if not r["skipped"]]
    y_true = [r["true_mode"] for r in valid_records]
    y_pred = [classify_mode(r["B6"]["text"]) for r in valid_records]

    summary = {
        "acc":        acc(y_true, y_pred)          if y_true else 0.0,
        "macro_f1":   macro_f1(y_true, y_pred)     if y_true else 0.0,
        "FAR":        far_refuse_conflict(y_true, y_pred) if y_true else 0.0,
        "substring":  answer_substring(valid_records, y_true),
        "counts":     dict(Counter(y_pred)),
        "n_evaluated": len(valid_records),
        "n_skipped":   len(skipped),
        "n_total":     len(records),
    }

    errors = [
        {
            "id":           r["id"],
            "true_mode":    t,
            "pred_mode":    p,
            "gold_answer":  r.get("gold_answer", ""),
            "answer_field": extract_json_answer(r["B6"]["text"]) or "",
            "full_text":    r["B6"]["text"],
        }
        for r, t, p in zip(valid_records, y_true, y_pred) if t != p
    ]

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file  = out_dir / f"baseline_b6_{args.split}.json"
    err_file  = out_dir / f"baseline_b6_{args.split}_errors.json"
    skip_file = out_dir / f"baseline_b6_{args.split}_skipped.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": records}, f,
                  indent=2, ensure_ascii=False)
    with open(err_file, "w", encoding="utf-8") as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    with open(skip_file, "w", encoding="utf-8") as f:
        json.dump(skipped, f, indent=2, ensure_ascii=False)

    print(f"\n[{time.strftime('%H:%M:%S')}] Summary: {summary}")
    print(f"Saved {len(records)} records      -> {out_file}")
    print(f"Saved {len(errors)} errors        -> {err_file}")
    print(f"Saved {len(skipped)} skipped ids  -> {skip_file}")
    if oom_count:
        print(f"⚠️  {oom_count} instances skipped due to OOM — consider lowering --max_input_tokens")

if __name__ == "__main__":
    main()
