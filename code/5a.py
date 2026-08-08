import argparse, json, numpy as np, gc, time
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompts import render_prompt, DECODE_CFG, REFUSE_STRING, CONFLICT_STRING

# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--split",       default="test", choices=["train","val","test"])
parser.add_argument("--model",       default="ibm-granite/granite-3.1-8b-instruct")
parser.add_argument("--max_entries", type=int, default=None)
parser.add_argument("--out_dir",     default="results")
parser.add_argument("--thresholds",  default="b3_thresholds.json",
                    help="JSON file containing tuned thresholds from val.")
parser.add_argument("--max_new_tokens", type=int, default=None,
                    help="Override DECODE_CFG['max_new_tokens'] if provided.")
args = parser.parse_args()

RESULTS_DIR = Path(args.out_dir)
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────────────────────────────────────
try:
    with open(args.thresholds, "r", encoding="utf-8") as f:
        b3_cfg = json.load(f)
    B3_TOP1   = b3_cfg["top1_thresh"]
    B3_GAP    = b3_cfg["gap_thresh"]
    B3_ENT    = b3_cfg["ent_thresh"]
    B4_NLL_TH = b3_cfg.get("b4_nll_thresh", 3.5)
    print(f"Thresholds loaded: top1>{B3_TOP1} gap>{B3_GAP} ent>{B3_ENT} nll>{B4_NLL_TH}")
except FileNotFoundError:
    B3_TOP1, B3_GAP, B3_ENT, B4_NLL_TH = 0.5, 0.15, 0.9, 3.5
    print("⚠️  b3_thresholds.json not found — using defaults. Run val pass first.")

# ──────────────────────────────────────────────────────────────────────────────
# Load instances
# ──────────────────────────────────────────────────────────────────────────────
instances = []
with open(f"instances_{args.split}.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        instances.append(json.loads(line))

if args.max_entries:
    instances = instances[:args.max_entries]
print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(instances)} instances ({args.split})")

# ──────────────────────────────────────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────────────────────────────────────
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

def get_input_device(m):
    """
    device_map="auto" safe: place inputs on the device hosting embeddings.
    """
    if hasattr(m, "hf_device_map") and isinstance(m.hf_device_map, dict):
        for k in ["model.embed_tokens", "model.model.embed_tokens", "transformer.wte"]:
            if k in m.hf_device_map:
                return torch.device(m.hf_device_map[k])
        # fallback to any device in map
        return torch.device(next(iter(m.hf_device_map.values())))
    return next(m.parameters()).device

INPUT_DEVICE = get_input_device(model)
print(f"[{time.strftime('%H:%M:%S')}] Model loaded. input_device={INPUT_DEVICE}")

# decoding config (frozen; allow max_new_tokens override)
MAX_NEW = args.max_new_tokens if args.max_new_tokens is not None else DECODE_CFG["max_new_tokens"]

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def to_device(batch):
    return {k: v.to(INPUT_DEVICE) for k, v in batch.items()}

def generate(prompt: str) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = to_device(enc)
    inp_len = enc["input_ids"].shape[1]

    with torch.inference_mode():
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW,
            do_sample=DECODE_CFG["do_sample"],
            temperature=DECODE_CFG["temperature"],
            top_p=DECODE_CFG["top_p"],
            repetition_penalty=DECODE_CFG["repetition_penalty"],
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0][inp_len:], skip_special_tokens=True).strip()
    del enc, out
    return text

def compute_uncertainty_stats(prompt: str, answer: str) -> dict:
    """
    Teacher-forcing NLL + mean max-prob + mean margin.
    Uses correct device placement for device_map models.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(INPUT_DEVICE)
    answer_ids = tokenizer(answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(INPUT_DEVICE)

    if answer_ids.shape[1] == 0:
        return {"nll": float("inf"), "mean_max_prob": 0.0, "mean_margin": 0.0}

    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    prompt_len = prompt_ids.shape[1]

    with torch.inference_mode():
        logits = model(input_ids=full_ids).logits[0].float()  # [full_len, vocab]

    # logits predicting each answer token
    pred_logits = logits[prompt_len - 1: prompt_len + answer_ids.shape[1] - 1]  # [ans_len, vocab]
    targets = answer_ids[0]                                                     # [ans_len]

    probs = torch.softmax(pred_logits, dim=-1)
    top2 = torch.topk(probs, k=2, dim=-1).values
    nll = torch.nn.functional.cross_entropy(pred_logits, targets).item()

    del prompt_ids, answer_ids, full_ids, logits, pred_logits, probs, top2
    return {
        "nll": float(nll),
        "mean_max_prob": float(top2[:, 0].mean().item()) if 'top2' in locals() else 0.0,
        "mean_margin": float((top2[:, 0] - top2[:, 1]).mean().item()) if 'top2' in locals() else 0.0,
    }

def softmax_scores(scores):
    s = np.array(scores, dtype=np.float64)
    if s.size == 0:
        return np.array([], dtype=np.float64)
    s = s - s.max()
    e = np.exp(s)
    return e / e.sum()

# Unified triage instruction (B2)
TRIAGE_INSTR = (
    "You must use ONLY the provided documents. Do NOT use outside knowledge.\n"
    f"If the documents lack sufficient information, respond exactly: {REFUSE_STRING}\n"
    f"If the documents contain conflicting information, respond exactly: {CONFLICT_STRING}\n"
    "Otherwise, answer directly using only the documents.\n"
    "If documents conflict, do NOT choose a side and do NOT guess."
)

# ──────────────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────────────
records = []

for idx, inst in enumerate(tqdm(instances, desc=f"baselines/{args.split}")):
    q = inst["question"]
    docs = inst["docs"]
    scores = inst.get("retriever_scores", [])

    rec = {
        "id": inst["id"],
        "true_mode": inst.get("true_mode"),
        "gold_answer": inst.get("gold_answer", ""),
    }

    # B0 Raw baseline (no system grounding) — P3
    rec["B0"] = generate(render_prompt("P3", q, docs, tokenizer=tokenizer))

    # B1 Always answer grounded
    rec["B1"] = generate(render_prompt(
        "P0", q, docs,
        instruction="Answer the question directly using only the documents.",
        tokenizer=tokenizer
    ))

    # B2 Prompt-only triage (unified triage prompt)
    rec["B2"] = generate(render_prompt(
        "P0", q, docs,
        instruction=TRIAGE_INSTR,
        tokenizer=tokenizer
    ))

    # B3 Retriever gating (rule-based)
    p = softmax_scores(scores)
    if p.size == 0:
        top1, gap, ent = 0.0, 0.0, 0.0
        rec["B3"] = REFUSE_STRING  # no scores → abstain
    else:
        p_sort = sorted(p, reverse=True)
        top1 = float(p_sort[0])
        gap = float(p_sort[0] - p_sort[1]) if len(p_sort) > 1 else top1
        ent = float(-np.sum(p * np.log(p + 1e-9)))

        if ent > B3_ENT:
            rec["B3"] = CONFLICT_STRING
        elif top1 < B3_TOP1 or gap < B3_GAP:
            rec["B3"] = REFUSE_STRING
        else:
            rec["B3"] = generate(render_prompt(
                "P0", q, docs,
                instruction="Answer the question directly using only the documents.",
                tokenizer=tokenizer
            ))

    rec["B3_top1"] = top1
    rec["B3_gap"] = gap
    rec["B3_ent"] = ent

    # B4 Uncertainty gating via self NLL of generated answer
    ans_prompt = render_prompt(
        "P0", q, docs,
        instruction="Answer the question using only the documents.",
        tokenizer=tokenizer
    )
    ans_text = generate(ans_prompt)
    unc = compute_uncertainty_stats(ans_prompt, ans_text)

    rec["B4_nll"] = unc["nll"]
    rec["B4_mean_max_prob"] = unc["mean_max_prob"]
    rec["B4_mean_margin"] = unc["mean_margin"]
    rec["B4_answer"] = ans_text
    rec["B4"] = REFUSE_STRING if unc["nll"] > B4_NLL_TH else ans_text

    records.append(rec)

    if (idx + 1) % 20 == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

out_file = RESULTS_DIR / f"baselines_{args.split}.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(records, f, indent=2, ensure_ascii=False)

print(f"[{time.strftime('%H:%M:%S')}] Saved {len(records)} records → {out_file}")
