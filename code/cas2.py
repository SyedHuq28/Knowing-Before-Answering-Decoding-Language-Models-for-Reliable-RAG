import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from warnings import catch_warnings, simplefilter

import numpy as np
import torch
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from prompts import render_prompt

LABEL_INV = {0: "answer", 1: "refuse", 2: "conflict"}
PATCH_DIRECTIONS = [
    ("refuse", "answer"),
    ("answer", "refuse"),
    ("conflict", "answer"),
    ("answer", "conflict"),
]


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


def load_best_layer(probe_results_path: Path, metric: str = "val_acc") -> int:
    with open(probe_results_path, "r") as f:
        data = json.load(f)
    best = max(data.items(), key=lambda kv: kv[1][metric])
    best_layer = int(best[0])
    print(f"Best probe layer: L{best_layer} ({metric}={best[1][metric]:.4f})")
    return best_layer


class TorchLayerRouters:
    def __init__(self, means, scales, weights, biases, class_ids, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.means = [m.to(device=device, dtype=dtype) for m in means]
        self.scales = [s.to(device=device, dtype=dtype) for s in scales]
        self.weights = [w.to(device=device, dtype=dtype) for w in weights]
        self.biases = [b.to(device=device, dtype=dtype) for b in biases]
        self.class_ids = [c.to(device=device, dtype=torch.long) for c in class_ids]
        self.num_layers = len(self.weights)

    @torch.inference_mode()
    def predict_layer_id(self, layer_idx: int, x: torch.Tensor) -> int:
        x = x.to(device=self.device, dtype=self.dtype)
        x = (x - self.means[layer_idx]) / self.scales[layer_idx]
        logits = x @ self.weights[layer_idx].T + self.biases[layer_idx]
        pred_pos = int(torch.argmax(logits).item())
        pred_id = int(self.class_ids[layer_idx][pred_pos].item())
        return pred_id

    @torch.inference_mode()
    def route_all_layers(self, hidden_states, L: int) -> List[str]:
        preds = []
        for l in range(L):
            vec = hidden_states[l + 1][0, -1, :]
            pred_id = self.predict_layer_id(l, vec)
            preds.append(LABEL_INV[pred_id])
        return preds


def load_per_layer_routers_gpu(
    act_dir: Path,
    template: str,
    torch_device: torch.device,
) -> Tuple[TorchLayerRouters, int]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    H_tr = np.load(act_dir / f"H_{template}_train.npy").astype(np.float32)
    y_tr = np.load(act_dir / f"y_{template}_train.npy")
    L = H_tr.shape[1]

    print(f"[{time.strftime('%H:%M:%S')}] Fitting per-layer routers ({L} layers) ...")

    means, scales, weights, biases, class_ids = [], [], [], [], []

    for l in range(L):
        sc = StandardScaler()
        X = sc.fit_transform(H_tr[:, l, :])

        clf = LogisticRegression(
            max_iter=4000,
            tol=1e-4,
            C=1.0,
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )

        had_warning = False
        with catch_warnings(record=True) as w:
            simplefilter("always", ConvergenceWarning)
            clf.fit(X, y_tr)
            had_warning = any(issubclass(wi.category, ConvergenceWarning) for wi in w)

        if had_warning:
            print(f"[warn] L{l}: lbfgs did not converge, retrying with saga")
            clf = LogisticRegression(
                max_iter=6000,
                tol=1e-4,
                C=1.0,
                multi_class="multinomial",
                solver="saga",
                random_state=42,
                n_jobs=-1,
            )
            clf.fit(X, y_tr)

        mean = torch.from_numpy(sc.mean_.astype(np.float32))
        scale = torch.from_numpy(sc.scale_.astype(np.float32))
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)

        W = torch.from_numpy(clf.coef_.astype(np.float32))
        b = torch.from_numpy(clf.intercept_.astype(np.float32))
        cls = torch.from_numpy(clf.classes_.astype(np.int64))

        means.append(mean)
        scales.append(scale)
        weights.append(W)
        biases.append(b)
        class_ids.append(cls)

    routers = TorchLayerRouters(
        means=means,
        scales=scales,
        weights=weights,
        biases=biases,
        class_ids=class_ids,
        device=torch_device,
        dtype=torch.float32,
    )

    print(f"[{time.strftime('%H:%M:%S')}] Per-layer GPU routers ready.")
    return routers, L


def forward_pass(
    model,
    tokenizer,
    blocks,
    input_dev: torch.device,
    inst: dict,
    template: str,
    patch_layer: Optional[int] = None,
    patch_vec: Optional[torch.Tensor] = None,
    max_docs: Optional[int] = None,
):
    docs = normalize_docs(inst["docs"])
    if max_docs:
        docs = docs[:max_docs]

    prompt = render_prompt(template, inst["question"], docs, tokenizer=tokenizer)
    enc = {k: v.to(input_dev) for k, v in tokenizer(prompt, return_tensors="pt").items()}

    handle = [None]

    if patch_layer is not None and patch_vec is not None:
        def hook(_mod, _inp, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            if not torch.is_tensor(h) or h.ndim != 3:
                return None
            h2 = h.clone()
            h2[0, -1, :] = patch_vec.to(device=h2.device, dtype=h2.dtype)
            return (h2,) + output[1:] if is_tuple else h2

        handle[0] = blocks[patch_layer].register_forward_hook(hook)

    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)

    if handle[0] is not None:
        handle[0].remove()

    hs = tuple(h.detach() for h in out.hidden_states)
    del enc, out
    return hs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe_results", default="results/layer_probe_hidden.json")
    ap.add_argument("--act_dir", default="activations")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--template", default="P0")
    ap.add_argument("--probe_metric", default="val_acc", choices=["val_acc", "val_f1"])
    ap.add_argument("--n_pairs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--router_device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    act_dir = Path(args.act_dir)

    patch_layer = load_best_layer(Path(args.probe_results), args.probe_metric)

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

    blocks = get_layers(model)
    L = len(blocks)
    input_dev = get_input_device(model)

    if args.router_device == "auto":
        router_device = input_dev if input_dev.type == "cuda" else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        router_device = torch.device(args.router_device)

    print(
        f"[{time.strftime('%H:%M:%S')}] Model loaded. "
        f"L={L} patch_layer=L{patch_layer} input_dev={input_dev} router_dev={router_device}"
    )

    routers, L_cached = load_per_layer_routers_gpu(act_dir, args.template, router_device)
    assert L_cached == L, f"Cached L={L_cached} != model L={L}"

    inst_by_mode: Dict[str, List] = {"answer": [], "refuse": [], "conflict": []}
    with open(f"instances_{args.split}.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            inst = json.loads(line)
            inst_by_mode[inst["true_mode"]].append(inst)

    print({k: len(v) for k, v in inst_by_mode.items()})

    obs_layers = list(range(patch_layer, L))
    print(f"Observing {len(obs_layers)} layers: L{patch_layer} ... L{L-1}")

    all_results = {}

    for src_mode, tgt_mode in PATCH_DIRECTIONS:
        srcs = inst_by_mode[src_mode]
        tgts = inst_by_mode[tgt_mode]
        if not srcs or not tgts:
            print(f"Skipping {src_mode}->{tgt_mode}: empty pool")
            continue

        n = min(args.n_pairs, len(srcs), len(tgts))
        src_sel = random.sample(srcs, n)
        tgt_sel = random.sample(tgts, n)

        baseline_src_counts = np.zeros(len(obs_layers), dtype=np.float32)
        patched_src_counts = np.zeros(len(obs_layers), dtype=np.float32)
        records = []

        for i in tqdm(range(n), desc=f"{src_mode}->{tgt_mode}", leave=False):
            src = src_sel[i]
            tgt = tgt_sel[i]

            hs_base = forward_pass(
                model, tokenizer, blocks, input_dev,
                tgt, args.template, max_docs=args.max_docs,
            )
            base_preds = routers.route_all_layers(hs_base, L)
            del hs_base

            hs_src = forward_pass(
                model, tokenizer, blocks, input_dev,
                src, args.template, max_docs=args.max_docs,
            )
            src_vec = hs_src[patch_layer + 1][0, -1, :].detach()
            del hs_src

            hs_patched = forward_pass(
                model, tokenizer, blocks, input_dev,
                tgt, args.template,
                patch_layer=patch_layer,
                patch_vec=src_vec,
                max_docs=args.max_docs,
            )
            patched_preds = routers.route_all_layers(hs_patched, L)
            del hs_patched, src_vec

            for j, obs_l in enumerate(obs_layers):
                if base_preds[obs_l] == src_mode:
                    baseline_src_counts[j] += 1
                if patched_preds[obs_l] == src_mode:
                    patched_src_counts[j] += 1

            records.append({
                "pair_idx": i,
                "src_id": src["id"],
                "tgt_id": tgt["id"],
                "baseline_obs": [base_preds[l] for l in obs_layers],
                "patched_obs": [patched_preds[l] for l in obs_layers],
                "flipped_at_patch": (patched_preds[patch_layer] == src_mode),
            })

            if (i + 1) % 10 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

        baseline_rate = (baseline_src_counts / max(1, n)).tolist()
        patched_rate = (patched_src_counts / max(1, n)).tolist()
        delta = [p - b for p, b in zip(patched_rate, baseline_rate)]
        flip_at_patch = float(sum(r["flipped_at_patch"] for r in records) / max(1, n))

        direction_key = f"{src_mode}_to_{tgt_mode}"
        all_results[direction_key] = {
            "patch_layer": patch_layer,
            "obs_layers": obs_layers,
            "n_pairs": n,
            "src_mode": src_mode,
            "tgt_mode": tgt_mode,
            "flip_at_patch_layer": flip_at_patch,
            "baseline_source_rate": baseline_rate,
            "patched_source_rate": patched_rate,
            "delta": delta,
            "records": records,
        }

        peak_l = obs_layers[int(np.argmax(delta))]
        print(
            f"  {src_mode}->{tgt_mode}: "
            f"flip@L{patch_layer}={flip_at_patch:.3f} "
            f"peak_delta={max(delta):.3f} @ obs_L{peak_l}"
        )

    payload = {
        "meta": {
            "model": args.model,
            "split": args.split,
            "template": args.template,
            "patch_layer": patch_layer,
            "obs_layers": obs_layers,
            "n_pairs": args.n_pairs,
            "L": L,
            "router_device": str(router_device),
        },
        "results": all_results,
    }

    full_path = out_dir / f"causal_tracing_{args.split}.json"
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Full results → {full_path}")

    compact = {
        "meta": payload["meta"],
        "results": {
            k: {sk: sv for sk, sv in v.items() if sk != "records"}
            for k, v in all_results.items()
        },
    }
    compact_path = out_dir / f"causal_tracing_{args.split}_compact.json"
    with open(compact_path, "w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)
    print(f"[{time.strftime('%H:%M:%S')}] Compact → {compact_path}")


if __name__ == "__main__":
    main()
