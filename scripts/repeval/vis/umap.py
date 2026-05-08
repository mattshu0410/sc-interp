"""Per-condition UMAP on captured activations using `torchdr.UMAP`."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torchdr import UMAP

from scripts.repeval.vis._common import auto_name, run_vis

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "predictions" / "repeval" / "vis" / "umap"
DEFAULT_N_CELLS = 10_000


def _to_numpy(y) -> np.ndarray:
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    return np.asarray(y)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m scripts.repeval.vis.umap")
    p.add_argument("--conditions", required=True)
    p.add_argument("--layer", required=True)
    p.add_argument("--phase", default="predict")
    p.add_argument("--color-by", default="cell_line", choices=["cell_line", "pert"])
    p.add_argument("--n-cells", type=int, default=DEFAULT_N_CELLS)
    p.add_argument("--n-neighbors", type=int, default=15)
    p.add_argument("--min-dist", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--force", action="store_true",
                   help="refit even if cached <cond>.npz exists in out_dir")
    args = p.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    out_dir = args.output or (DEFAULT_OUT_ROOT / (args.name or auto_name(conditions, args.layer)))

    def fit_fn(x: np.ndarray) -> np.ndarray:
        model = UMAP(
            n_components=2,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            max_iter=args.max_iter,
            random_state=args.seed,
            device=args.device,
            verbose=False,
        )
        return _to_numpy(model.fit_transform(x.astype(np.float32)))

    run_vis(
        method_tag="vis/umap",
        fit_fn=fit_fn,
        conditions=conditions,
        layer=args.layer,
        phase=args.phase,
        color_by=args.color_by,
        n_cells=args.n_cells,
        out_dir=out_dir,
        extra_meta={
            "implementation": "torchdr.UMAP",
            "n_neighbors": args.n_neighbors,
            "min_dist": args.min_dist,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "device": args.device,
        },
        force=args.force,
    )


if __name__ == "__main__":
    main()
