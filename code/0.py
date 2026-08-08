import json

INPUT     = "rcsr_processed.json"
MODES     = ["answer", "refuse", "conflict"]
LABEL_MAP = {"answer": 0, "refuse": 1, "conflict": 2}

with open(INPUT) as f:
    data = json.load(f)

splits = {"train": [], "val": [], "test": []}

for entry in data:
    split = entry["split"]
    for mode in MODES:
        docs   = entry["mode_bundles"].get(mode, [])
        if not docs:
            continue

        raw_scores = entry["retriever_scores"].get(mode, [])

        # Fix 1: ensure scores align with docs — truncate or pad with 0.5
        if len(raw_scores) > len(docs):
            scores = raw_scores[:len(docs)]
        elif len(raw_scores) < len(docs):
            scores = raw_scores + [0.5] * (len(docs) - len(raw_scores))
        else:
            scores = raw_scores

        assert len(scores) == len(docs), "Score/doc mismatch after alignment"

        instance = {
            "id":               f"{entry['id']}_{mode}",
            "original_id":      entry["id"],
            "question":         entry["question"],
            "gold_answer":      entry.get("answer", ""),
            "docs":             docs,
            "retriever_scores": scores,
            "true_mode":        mode,
            "label":            LABEL_MAP[mode],
            "split":            split,
        }
        splits[split].append(instance)

for split, rows in splits.items():
    out = f"instances_{split}.jsonl"
    with open(out, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"{split}: {len(rows)} instances → {out}")

label_policy = {
    "Answer":   "At least one retrieved document contains a span directly supporting the gold answer.",
    "Refuse":   "No retrieved document contains sufficient evidence to derive the gold answer.",
    "Conflict": "At least two retrieved documents make mutually incompatible factual claims.",
}
with open("label_policy.json", "w") as f:
    json.dump(label_policy, f, indent=2)
print("label_policy.json saved.")

