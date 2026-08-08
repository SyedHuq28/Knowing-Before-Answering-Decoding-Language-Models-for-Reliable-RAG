#!/usr/bin/env python3
import argparse, json, glob, re
from pathlib import Path
from collections import Counter
import numpy as np

REFUSE_STRING   = "Not enough information."
CONFLICT_STRING = "Documents contain conflicting information."

def first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""

# Only look at FIRST LINE for mode decision
# Refuse cues (first-line)
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

# Conflict cues (first-line). Require doc/source mention OR canonical string.
CONFLICT_RE = re.compile(
    r"\b("
    r"documents?|sources?|passages?|provided documents?"
    r")\b.*\b("
    r"conflict|conflicting|contradict|contradiction|inconsistent|disagree|cannot reconcile|at odds"
    r")\b",
    re.IGNORECASE,
)

def classify_mode(text: str) -> str:
    line = first_nonempty_line(text)
    low  = line.lower()

    # canonical exact-start signals
    if low.startswith(CONFLICT_STRING.lower()):
        return "conflict"
    if low.startswith(REFUSE_STRING.lower()):
        return "refuse"

    # first-line pattern fallback
    if CONFLICT_RE.search(line):
        return "conflict"
    if REFUSE_RE.search(line):
        return "refuse"
    return "answer"

def macro_f1(y_true, y_pred, labels=("answer","refuse","conflict")) -> float:
    f1s=[]
    for lab in labels:
        tp=sum((t==lab and p==lab) for t,p in zip(y_true,y_pred))
        fp=sum((t!=lab and p==lab) for t,p in zip(y_true,y_pred))
        fn=sum((t==lab and p!=lab) for t,p in zip(y_true,y_pred))
        prec=tp/(tp+fp+1e-9)
        rec =tp/(tp+fn+1e-9)
        f1 =2*prec*rec/(prec+rec+1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))

def substring_match(gold: str, pred: str) -> float:
    g=(gold or "").strip().lower()
    p=(pred or "").strip().lower()
    if not g:
        return 0.0
    return 1.0 if g in p else 0.0

def summarize(rows, cond):
    y_true=[r["true_mode"] for r in rows]
    y_pred=[classify_mode(r[cond]["text"]) for r in rows]

    acc=float(np.mean([t==p for t,p in zip(y_true,y_pred)]))
    mf1=macro_f1(y_true, y_pred)

    denom=sum(t in ("refuse","conflict") for t in y_true)
    far=sum((t in ("refuse","conflict") and p=="answer") for t,p in zip(y_true,y_pred))/max(1,denom)

    ans_idx=[i for i,t in enumerate(y_true) if t=="answer"]
    ans_sub=float(np.mean([substring_match(rows[i].get("gold_answer",""), rows[i][cond]["text"]) for i in ans_idx])) if ans_idx else 0.0

    counts=Counter(y_pred)
    return {"acc":acc,"macro_f1":mf1,"FAR_refuse+conflict":float(far),"answer_substring":ans_sub,"pred_counts":dict(counts)}

def rescore_file(path: str, out_path: str):
    obj=json.load(open(path,"r",encoding="utf-8"))
    rows=obj["rows"]
    out={
        "meta": obj.get("summary",{}).get("meta",{}),
        "S1": summarize(rows,"S1"),
        "S2": summarize(rows,"S2"),
        "S3": summarize(rows,"S3"),
        "S4": summarize(rows,"S4"),
    }
    json.dump(out, open(out_path,"w",encoding="utf-8"), indent=2)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/ablation_val_a*.json",
                    help="which ablation files to rescore")
    ap.add_argument("--out_dir", default="results/rescored")
    args=ap.parse_args()

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files=sorted(glob.glob(args.glob))
    if not files:
        raise SystemExit(f"No files matched: {args.glob}")

    all_summ={}
    for fp in files:
        out_fp=out_dir/(Path(fp).stem + "_rescored.json")
        summ=rescore_file(fp, str(out_fp))
        all_summ[Path(fp).name]=summ
        print(f"{Path(fp).name} -> {out_fp.name}")
        print("  S2:", summ["S2"])
        print("  S3:", summ["S3"])
        print("  S4:", summ["S4"])

    json.dump(all_summ, open(out_dir/"all_rescored.json","w",encoding="utf-8"), indent=2)
    print("\nSaved:", out_dir/"all_rescored.json")

if __name__=="__main__":
    main()