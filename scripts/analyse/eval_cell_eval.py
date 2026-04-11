"""
Score a prediction h5ad against ground truth using Arc's cell-eval.

Usage:
    source tools/.venv/bin/activate
    python -m scripts.analyse.eval_cell_eval \\
        --predictions predictions/scgpt_norman_test.h5ad \\
        --profile full

The prediction h5ad must be self-contained (produced by any run_*.py):
predicted expression in X, ground truth in layers["truth"], both
including control cells with label = uns["control_label"]. We split it
into the two AnnDatas cell-eval expects and call MetricsEvaluator.
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from cell_eval import MetricsEvaluator

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument(
        "--profile",
        default="full",
        choices=["full", "minimal", "vcc", "de", "anndata"],
    )
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument(
        "--outdir",
        type=Path,
        help="output dir, default eval_outputs/<predictions stem>/",
    )
    return p.parse_args()


def split_prediction_adata(
    adata: ad.AnnData,
) -> tuple[ad.AnnData, ad.AnnData, str, str]:
    pert_col = adata.uns["pert_col"]
    control_label = adata.uns["control_label"]

    def _to_dense(x):
        return x.toarray() if hasattr(x, "toarray") else np.asarray(x)

    obs = pd.DataFrame({pert_col: adata.obs[pert_col].values})
    obs.index = adata.obs_names.astype(str)

    # cell-eval requires non-negative log-normalised values. Some models
    # (scGPT) can emit tiny negative predictions from a linear decoder;
    # clip at zero here rather than in the canonical prediction h5ad.
    pred_X = np.clip(_to_dense(adata.X), 0, None)
    truth_X = _to_dense(adata.layers["truth"])

    adata_pred = ad.AnnData(X=pred_X, obs=obs, var=adata.var.copy())
    adata_real = ad.AnnData(X=truth_X, obs=obs.copy(), var=adata.var.copy())
    return adata_pred, adata_real, pert_col, control_label


def main() -> None:
    args = parse_args()
    adata = ad.read_h5ad(args.predictions)
    adata_pred, adata_real, pert_col, control_label = split_prediction_adata(adata)

    print(f"==> pred shape: {adata_pred.shape}")
    print(f"==> real shape: {adata_real.shape}")
    print(f"==> pert_col: {pert_col!r}, control_label: {control_label!r}")

    outdir = args.outdir or (REPO_ROOT / "eval_outputs" / args.predictions.stem)
    outdir.mkdir(parents=True, exist_ok=True)

    evaluator = MetricsEvaluator(
        adata_pred=adata_pred,
        adata_real=adata_real,
        control_pert=control_label,
        pert_col=pert_col,
        num_threads=args.num_threads,
        outdir=str(outdir),
        allow_discrete=False,
    )
    results, agg_results = evaluator.compute(profile=args.profile)

    print()
    print("=== aggregated metrics ===")
    print(agg_results)
    print()
    print(f"==> per-pert metrics and full outputs written to {outdir}")


if __name__ == "__main__":
    main()
