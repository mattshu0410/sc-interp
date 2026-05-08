"""Shared helpers for vis/ scripts: load + fit + plot pipeline."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.repeval import CaptureView


def auto_name(conditions: list[str], layer: str) -> str:
    layer_short = layer.rsplit(".", 1)[-1]
    return "+".join(conditions) + f"__L{layer_short}"


def plot_comparison(
    out_path: Path,
    embeddings: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    color_by: str,
    title_prefix: str,
) -> None:
    """One subplot per condition; consistent colour scheme across panels."""
    conditions = list(embeddings)
    n = len(conditions)

    all_labels = np.concatenate([labels[c] for c in conditions])
    levels = sorted(np.unique(all_labels).tolist())
    cmap = plt.get_cmap("tab10" if len(levels) <= 10 else "tab20")
    color = {lv: cmap(i % cmap.N) for i, lv in enumerate(levels)}

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    for ax, cond in zip(axes[0], conditions):
        y = embeddings[cond]
        labs = labels[cond]
        for lv in levels:
            mask = labs == lv
            if mask.sum() == 0:
                continue
            ax.scatter(y[mask, 0], y[mask, 1], s=3, alpha=0.25,
                       c=[color[lv]], label=str(lv), edgecolors="none")
        ax.set_title(f"{cond}  (n={len(y)})")
        ax.set_xticks([])
        ax.set_yticks([])

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=6,
                   color=color[lv], label=str(lv))
        for lv in levels
    ]
    axes[0, -1].legend(handles=handles, loc="upper right",
                       title=color_by, fontsize=8, markerscale=1)
    fig.suptitle(f"{title_prefix} (per-condition fit, coloured by {color_by})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_vis(
    *,
    method_tag: str,
    fit_fn: Callable[[np.ndarray], np.ndarray],
    conditions: list[str],
    layer: str,
    phase: str,
    color_by: str,
    n_cells: int,
    out_dir: Path,
    extra_meta: dict,
    force: bool = False,
) -> None:
    """Per-condition: load → fit_fn → save embedding + labels. Then comparison plot + meta.

    If a per-condition `.npz` already exists in `out_dir` and `force=False`,
    the cached embedding+labels are loaded and the fit is skipped. The
    comparison plot is always re-rendered (cheap, lets you change `color_by`
    or alpha without recomputing). Pass `force=True` (or delete the dir)
    to invalidate and refit.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> writing to {out_dir}")

    embeddings: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for cond in conditions:
        print(f"== {cond} ==")
        npz_path = out_dir / f"{cond}.npz"
        if npz_path.exists() and not force:
            d = np.load(npz_path, allow_pickle=True)
            y = d["embedding"]
            labs_all = {k: d[k] for k in d.files if k != "embedding"}
            print(f"  using cached {npz_path.name}  shape={y.shape}")
        else:
            cv = CaptureView(cond, layer=layer, phase=phase)
            x = cv.per_cell(n_cells=n_cells)
            labs_all = cv.labels(n_cells=n_cells)
            print(f"  fitting on {x.shape}...")
            y = fit_fn(x)
            if y.ndim != 2 or y.shape[1] != 2:
                raise ValueError(f"fit_fn returned shape {y.shape}; expected (n, 2)")
            np.savez(npz_path, embedding=y, **labs_all)
            del x
        if color_by not in labs_all:
            raise KeyError(f"label {color_by!r} not in cached labels: {list(labs_all)}")
        embeddings[cond] = y
        labels[cond] = labs_all[color_by]

    plot_path = out_dir / "comparison.png"
    plot_comparison(plot_path, embeddings, labels, color_by,
                    title_prefix=f"{method_tag} of {layer}")
    print(f"==> wrote {plot_path}")

    meta = {
        "metric": method_tag,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "conditions": conditions,
        "layer": layer,
        "phase": phase,
        "color_by": color_by,
        "n_cells": n_cells,
        **extra_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"==> wrote {out_dir / 'meta.json'}")
