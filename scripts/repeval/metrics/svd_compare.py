"""Cross-condition SVD / PCA comparison at one capture layer.

Runs sklearn PCA on a random subsample of token positions per condition
(activations are per-feature z-scored via the stats sidecar inside
CaptureView). Reports concentration metrics + a side-by-side cumulative
variance plot.

CLI:
    python -m scripts.repeval.metrics.svd_compare \\
        --conditions base,esm,random \\
        --layer transformer_encoder.layers.7
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from scripts.repeval import CaptureView

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "predictions" / "repeval" / "svd_compare"
DEFAULT_N_CELLS = 10_000
DEFAULT_SVD_POSITIONS = 50_000


def pca_metrics(x: np.ndarray, *, n_positions: int, seed: int = 7) -> dict:
    """sklearn PCA on a random token-position subsample of (n, T, d)."""
    flat = x.reshape(-1, x.shape[-1])
    rng = np.random.default_rng(seed)
    idx = rng.choice(flat.shape[0], size=min(n_positions, flat.shape[0]), replace=False)
    sample = flat[idx].astype(np.float32)

    pca = PCA(n_components=sample.shape[1], svd_solver="full")
    pca.fit(sample)

    var = pca.explained_variance_                # (d,)
    var_ratio = pca.explained_variance_ratio_    # (d,)
    cum = np.cumsum(var_ratio)
    pr = float((var.sum() ** 2) / (var ** 2).sum())
    return {
        "explained_variance": var,
        "explained_variance_ratio": var_ratio,
        "cumulative_variance": cum,
        "trace": float(var.sum()),
        "top1": float(var[0]),
        "top10": float(var[9]),
        "participation_ratio": pr,
        "dims_90": int(np.searchsorted(cum, 0.90) + 1),
        "dims_99": int(np.searchsorted(cum, 0.99) + 1),
        "n_samples": int(sample.shape[0]),
    }


def auto_name(conditions: list[str], layer: str) -> str:
    layer_short = layer.rsplit(".", 1)[-1]
    return "+".join(conditions) + f"__L{layer_short}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scripts.repeval.metrics.svd_compare")
    p.add_argument("--conditions", required=True,
                   help="comma-separated condition names (e.g. base,esm,random)")
    p.add_argument("--layer", required=True,
                   help="full capture name (e.g. transformer_encoder.layers.7)")
    p.add_argument("--phase", default="predict")
    p.add_argument("--n-cells", type=int, default=DEFAULT_N_CELLS)
    p.add_argument("--svd-positions", type=int, default=DEFAULT_SVD_POSITIONS)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=None,
                   help=f"output dir; default {DEFAULT_OUT_ROOT}/<auto-name>/")
    p.add_argument("--name", type=str, default=None,
                   help="override the auto-derived comparison folder name")
    args = p.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    out_dir = args.output or (DEFAULT_OUT_ROOT / (args.name or auto_name(conditions, args.layer)))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> writing to {out_dir}")

    results: dict[str, dict] = {}
    for cond in conditions:
        print(f"== {cond} ==")
        cv = CaptureView(cond, layer=args.layer, phase=args.phase)
        x = cv.per_token(n_cells=args.n_cells)
        m = pca_metrics(x, n_positions=args.svd_positions, seed=args.seed)
        results[cond] = m
        print(f"  trace={m['trace']:.1f}  top1={m['top1']:.2f}  PR={m['participation_ratio']:.2f}  "
              f"dims@90%={m['dims_90']}  dims@99%={m['dims_99']}  (n_samples={m['n_samples']})")
        del x

    # summary.csv
    summary_path = out_dir / "summary.csv"
    cols = ["condition", "trace", "top1", "top10", "participation_ratio", "dims_90", "dims_99", "n_samples"]
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for cond in conditions:
            r = results[cond]
            w.writerow([cond] + [r[k] for k in cols[1:]])
    print(f"==> wrote {summary_path}")

    # Per-condition arrays.
    for cond in conditions:
        np.savez(out_dir / f"{cond}.npz",
                 explained_variance=results[cond]["explained_variance"],
                 explained_variance_ratio=results[cond]["explained_variance_ratio"],
                 cumulative_variance=results[cond]["cumulative_variance"])

    # Plot.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("tab10")
    color = {c: cmap(i) for i, c in enumerate(conditions)}

    ax = axes[0]
    for cond in conditions:
        r = results[cond]
        ax.plot(np.arange(1, len(r["explained_variance"]) + 1), r["explained_variance"],
                color=color[cond], alpha=0.85,
                label=f"{cond} (PR={r['participation_ratio']:.2f})")
    ax.set_yscale("log")
    ax.set_xlabel("component index")
    ax.set_ylabel("explained variance (log)")
    ax.set_title(f"{args.layer} PCA spectrum")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for cond in conditions:
        r = results[cond]
        ax.plot(np.arange(1, len(r["cumulative_variance"]) + 1), r["cumulative_variance"],
                color=color[cond], alpha=0.85,
                label=f"{cond} (90%@{r['dims_90']}, 99%@{r['dims_99']})")
    ax.axhline(0.90, color="k", linestyle="--", alpha=0.3)
    ax.axhline(0.99, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("# top components")
    ax.set_ylabel("cumulative variance ratio")
    ax.set_title("cumulative explained variance")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = out_dir / "comparison.png"
    plt.savefig(plot_path, dpi=120)
    plt.close(fig)
    print(f"==> wrote {plot_path}")

    # meta.json
    meta = {
        "metric": "svd_compare",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conditions": conditions,
        "layer": args.layer,
        "phase": args.phase,
        "n_cells": args.n_cells,
        "svd_positions": args.svd_positions,
        "seed": args.seed,
        "implementation": "sklearn.decomposition.PCA(svd_solver='full')",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"==> wrote {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
