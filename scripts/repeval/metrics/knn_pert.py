"""kNN perturbation retrieval (Bendidi 2024 §3.4) — cross-cell-line split.

For each condition, the latent space is built from one cell line (reference)
and queried with the other cell line (query). kNN with weighted-cosine
voting (Bendidi's recipe, ported from valence-labs/Tx-Evaluation):

    sim(q, r) = q.T r,   weight = exp(sim / T),   T = 0.07
    predicted pert = argmax_class sum_{neighbours of class} weight

Higher cross-cell-line top-k accuracy ⇒ pert representation transfers across
cell lines. With ``--within-baseline``, also reports an in-cell-line random
50/50 split as a ceiling.

CLI:
    python -m scripts.repeval.metrics.knn_pert \\
        --conditions base,esm,random \\
        --layer transformer_encoder.layers.7 \\
        --within-baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.repeval import CaptureView

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "predictions" / "repeval" / "knn_pert"


def _odd(k: int) -> int:
    return k if k % 2 else k + 1


def _device(arg: str) -> str:
    if arg != "auto":
        return arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def weighted_knn(
    ref_X: np.ndarray,
    ref_y: np.ndarray,
    query_X: np.ndarray,
    query_y: np.ndarray,
    *,
    k: int,
    temperature: float,
    device: str,
) -> tuple[float, float]:
    """Cosine + temperature-weighted majority vote. Returns (top1, top5).

    Built on torch so it runs on GPU when available; both reference and
    query embeddings move to the same device, similarity is computed as
    a single matmul, top-k via torch.topk."""
    label_to_idx = {l: i for i, l in enumerate(
        np.unique(np.concatenate([ref_y, query_y]))
    )}
    n_classes = len(label_to_idx)

    ref_t = torch.from_numpy(ref_X.astype(np.float32)).to(device)
    query_t = torch.from_numpy(query_X.astype(np.float32)).to(device)
    ref_t = torch.nn.functional.normalize(ref_t, dim=1)
    query_t = torch.nn.functional.normalize(query_t, dim=1)
    ref_idx = torch.tensor([label_to_idx[l] for l in ref_y],
                           dtype=torch.long, device=device)
    query_idx = torch.tensor([label_to_idx[l] for l in query_y],
                             dtype=torch.long, device=device)

    sims = query_t @ ref_t.T                       # (n_query, n_ref)
    top_k_sims, top_k_pos = sims.topk(k, dim=1, sorted=True)
    weights = (top_k_sims / temperature).exp()      # (n_query, k)
    neighbour_labels = ref_idx[top_k_pos]           # (n_query, k)

    n_query = query_idx.shape[0]
    class_weights = torch.zeros(
        (n_query, n_classes), device=device, dtype=weights.dtype
    )
    class_weights.scatter_add_(1, neighbour_labels, weights)

    pred_order = class_weights.argsort(dim=1, descending=True)
    top1 = (pred_order[:, 0] == query_idx).float().mean().item()
    top5 = (pred_order[:, :5] == query_idx[:, None]).any(dim=1).float().mean().item()
    return top1, top5


def _within_split(
    cell_line: np.ndarray, pert: np.ndarray, line: str, *, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Random ~50/50 split of `line` cells, stratified per (line, pert)."""
    rng = np.random.default_rng(seed)
    in_line = np.where(cell_line == line)[0]
    by_pert: dict[str, list[int]] = {}
    for i in in_line:
        by_pert.setdefault(pert[i], []).append(int(i))
    ref, query = [], []
    for plist in by_pert.values():
        plist = list(plist)
        rng.shuffle(plist)
        half = len(plist) // 2
        ref.extend(plist[:half])
        query.extend(plist[half:])
    return np.asarray(ref), np.asarray(query)


def auto_name(conditions: list[str], layer: str, pool: str) -> str:
    layer_short = layer.rsplit(".", 1)[-1]
    return "+".join(conditions) + f"__L{layer_short}__{pool}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scripts.repeval.metrics.knn_pert")
    p.add_argument("--conditions", required=True,
                   help="comma-separated condition names")
    p.add_argument("--layer", required=True,
                   help="full capture name (e.g. transformer_encoder.layers.7)")
    p.add_argument("--phase", default="predict")
    p.add_argument("--pool", default="mean", choices=["mean", "max"],
                   help="token aggregation: mean or element-wise max (default mean)")
    p.add_argument("--n-cells", type=int, default=None,
                   help="cap cells per condition; default uses full capture")
    p.add_argument("--reference-line", default="k562")
    p.add_argument("--query-line", default="rpe1")
    p.add_argument("--within-baseline", action="store_true",
                   help="also run in-cell-line random 50/50 split as a ceiling")
    p.add_argument("--k", type=int, default=None,
                   help="neighbours per query; default ⌊√n_ref⌋ (odd)")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--name", type=str, default=None)
    args = p.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    out_dir = args.output or (DEFAULT_OUT_ROOT / (args.name or auto_name(conditions, args.layer, args.pool)))
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    print(f"==> writing to {out_dir}  (device={device})")

    # We collect rows into summary.csv and emit a single comparison plot.
    rows: list[dict] = []
    for cond in conditions:
        print(f"== {cond} ==")
        cv = CaptureView(cond, layer=args.layer, phase=args.phase)
        n = args.n_cells if args.n_cells is not None else cv.n_cells
        x = cv.per_cell(n_cells=n, pool=args.pool)
        labs = cv.labels(n_cells=n)
        cell_line = labs["cell_line"]
        pert = labs["pert"]
        print(f"  loaded {x.shape}  k562={int((cell_line == 'k562').sum())}  "
              f"rpe1={int((cell_line == 'rpe1').sum())}")

        # ── cross-cell-line directions ───────────────────────────────
        directions: list[tuple[str, str, str]] = [
            (f"{args.reference_line}_to_{args.query_line}", args.reference_line, args.query_line),
            (f"{args.query_line}_to_{args.reference_line}", args.query_line, args.reference_line),
        ]
        for tag, ref_line, query_line in directions:
            ref_mask = cell_line == ref_line
            query_mask = cell_line == query_line
            ref_X, ref_y = x[ref_mask], pert[ref_mask]
            query_X, query_y = x[query_mask], pert[query_mask]
            k = args.k if args.k is not None else _odd(int(math.sqrt(len(ref_X))))
            t1, t5 = weighted_knn(
                ref_X, ref_y, query_X, query_y,
                k=k, temperature=args.temperature, device=device,
            )
            rows.append(dict(
                condition=cond, direction=tag,
                top1=t1, top5=t5, k=k,
                n_ref=len(ref_X), n_query=len(query_X),
            ))
            print(f"  {tag:>15}: top1={t1:.4f}  top5={t5:.4f}  k={k}  "
                  f"(n_ref={len(ref_X)}, n_query={len(query_X)})")

        # ── within-cell-line baseline ────────────────────────────────
        if args.within_baseline:
            for line in (args.reference_line, args.query_line):
                ref_idx, query_idx = _within_split(cell_line, pert, line, seed=args.seed)
                ref_X, ref_y = x[ref_idx], pert[ref_idx]
                query_X, query_y = x[query_idx], pert[query_idx]
                k = args.k if args.k is not None else _odd(int(math.sqrt(len(ref_X))))
                t1, t5 = weighted_knn(
                    ref_X, ref_y, query_X, query_y,
                    k=k, temperature=args.temperature, device=device,
                )
                tag = f"{line}_to_{line}"
                rows.append(dict(
                    condition=cond, direction=tag,
                    top1=t1, top5=t5, k=k,
                    n_ref=len(ref_X), n_query=len(query_X),
                ))
                print(f"  {tag:>15}: top1={t1:.4f}  top5={t5:.4f}  k={k}  "
                      f"(n_ref={len(ref_X)}, n_query={len(query_X)})")
        del x, labs

    # summary.csv
    summary_path = out_dir / "summary.csv"
    cols = ["condition", "direction", "top1", "top5", "k", "n_ref", "n_query"]
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"==> wrote {summary_path}")

    # plot: grouped bars per direction, separate panels for top1 and top5
    directions = [r["direction"] for r in rows if r["condition"] == conditions[0]]
    fig, axes = plt.subplots(1, 2, figsize=(6 + 1.5 * len(directions), 5))
    width = 0.8 / max(1, len(conditions))
    cmap = plt.get_cmap("tab10")
    color = {c: cmap(i) for i, c in enumerate(conditions)}
    for ax, metric in zip(axes, ("top1", "top5")):
        for ci, cond in enumerate(conditions):
            vals = [next(r[metric] for r in rows if r["condition"] == cond and r["direction"] == d)
                    for d in directions]
            xs = np.arange(len(directions)) + ci * width - 0.4 + width / 2
            ax.bar(xs, vals, width=width, color=color[cond], label=cond, alpha=0.85)
        ax.set_xticks(np.arange(len(directions)))
        ax.set_xticklabels(directions, rotation=20, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(f"kNN {metric} accuracy")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"kNN perturbation retrieval at {args.layer}")
    fig.tight_layout()
    plot_path = out_dir / "comparison.png"
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"==> wrote {plot_path}")

    meta = {
        "metric": "knn_pert",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conditions": conditions,
        "layer": args.layer,
        "phase": args.phase,
        "pool": args.pool,
        "reference_line": args.reference_line,
        "query_line": args.query_line,
        "within_baseline": args.within_baseline,
        "k": args.k,
        "temperature": args.temperature,
        "seed": args.seed,
        "device": device,
        "n_cells": args.n_cells,
        "implementation": "torch + cosine kNN (Bendidi 2024 weighted-vote)",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"==> wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
