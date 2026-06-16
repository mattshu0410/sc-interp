"""Rank ALL features by per-pert activation correlation with eval gain.

Inverts the question of ``cli.build``: instead of starting from a
feature selection (top-K by Δnorm) and asking "where do these fire?",
this script walks all 4096 features and asks "for the perts where the
clean condition outperforms the corrupt condition most, which features
are most active?"

Lightweight: only per-pert × per-feature means are aggregated (no
per-gene, no per-cell-line, no top-cells). The existing `cli.build`
output is unchanged.

Usage:

    python -m scripts.feature_dict.cli.gain_attribute \\
        --tag layer_7_topk_x8_k32_lr1e-04_steps50000 \\
        --model-dir predictions/diff/scgpt_base_vs_esm_replogle/crosscoder/layer_7/topk_x8_k32_lr1e-04_steps50000 \\
        --capture transformer_encoder.layers.7 \\
        --base predictions/scgpt-replogle-activations/base \\
        --esm  predictions/scgpt-replogle-activations/esm \\
        --primary-csv predictions/scgpt-replogle-esm-ft/eval/results.csv \\
        --baseline-csv base=predictions/scgpt-replogle-base-ft/eval/results.csv \\
        --top-k-perts 50
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats as scistats

from scripts.diffing.pairs import load_pair
from scripts.feature_dict.eval_join import join_eval_gains
from scripts.feature_dict.sources.crosscoder import CrosscoderSource
from scripts.feature_dict.walker import walk_pair


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "predictions" / "feature_dict"


def _parse_baseline(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(f"--baseline-csv expects LABEL=PATH, got {arg!r}")
    label, path = arg.split("=", 1)
    return label, Path(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--capture", type=str, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--esm", type=Path, required=True)
    p.add_argument("--primary-csv", type=Path, required=True)
    p.add_argument("--baseline-csv", type=_parse_baseline, action="append", required=True)
    p.add_argument("--gain-against", type=str, default=None)
    p.add_argument("--metric", type=str, default="pearson_delta")
    p.add_argument("--primary-label", type=str, default="esm")
    p.add_argument("--top-k-perts", type=int, default=50,
                   help="K for the 'top-K perts by gain' overrepresentation comparison.")
    p.add_argument("--phase", type=str, default="predict")
    p.add_argument("--chunk-rows", type=int, default=32)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--limit-cells", type=int, default=None)
    p.add_argument("--cell-line", type=str, default=None,
                   help="Restrict per-pert aggregation to cells of this line "
                        "(e.g. 'rpe1' for OOD-test-only evaluation). Default: all cells.")
    args = p.parse_args()

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tags = {"phase": args.phase}

    # 1) Eval gains
    print("==> joining eval gains")
    baselines = dict(args.baseline_csv)
    gains = join_eval_gains(
        primary_csv=args.primary_csv,
        primary_label=args.primary_label,
        baseline_csvs=baselines,
        metric=args.metric,
    )
    gain_label = args.gain_against or next(iter(baselines))
    gain_field = f"gain_vs_{gain_label}"
    if gain_field not in gains.columns:
        raise KeyError(f"--gain-against={gain_label!r} not in baselines {list(baselines)}")
    print(f"    {len(gains)} perts; gain column = {gain_field!r} (mean = {gains[gain_field].mean():+.3f})")

    # Top-K perts by gain (for the overrepresentation comparison)
    top_K_perts = set(gains[gain_field].nlargest(args.top_k_perts).index)
    print(f"    top {args.top_k_perts} perts by gain (sample): {sorted(top_K_perts)[:5]}...")

    # 2) Walk + per-pert per-feature aggregation
    pair = load_pair(args.base, args.esm, capture=args.capture, tags_a=tags, tags_b=tags)
    src = CrosscoderSource(model_dir=args.model_dir, device=device)
    F = src.dict_size
    print(f"    dict_size = {F}")

    # Per-pert: running sum (F,) and token count.
    pert_sum: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(F, dtype=np.float64))
    pert_count: dict[str, int] = defaultdict(int)

    cell_line_filter = args.cell_line
    if cell_line_filter:
        print(f"==> filtering to cells with cell_line == {cell_line_filter!r}")
    print("==> walking activations (per-pert per-feature means)...")
    t0 = time.time()
    n_cells = 0
    n_cells_kept = 0
    for i, chunk in enumerate(walk_pair(pair, src, chunk_rows=args.chunk_rows, device=device)):
        # codes (n, T, F) -> sum over (chunk-cells, tokens) per pert
        codes_cpu = chunk.codes.detach().to("cpu").to(torch.float32).numpy()
        n, T, _ = codes_cpu.shape
        n_cells += n
        if cell_line_filter:
            keep = chunk.cell_line == cell_line_filter
            if not keep.any():
                continue
            codes_cpu = codes_cpu[keep]
            chunk_pert = chunk.pert[keep]
            n_cells_kept += int(keep.sum())
        else:
            chunk_pert = chunk.pert
            n_cells_kept += n
        for pert_label in np.unique(chunk_pert):
            mask = chunk_pert == pert_label
            n_pert_cells = int(mask.sum())
            chunk_pert_sum = codes_cpu[mask].reshape(-1, F).sum(axis=0).astype(np.float64)
            pert_sum[str(pert_label)] += chunk_pert_sum
            pert_count[str(pert_label)] += n_pert_cells * T
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            kept_msg = f" ({n_cells_kept} kept)" if cell_line_filter else ""
            print(f"    chunk {i+1} | {n_cells} cells{kept_msg} | {elapsed:.1f}s ({n_cells/elapsed:.1f} cells/s)")
        if args.limit_cells and n_cells >= args.limit_cells:
            print(f"    --limit-cells={args.limit_cells} hit; stopping")
            break
    walk_elapsed = time.time() - t0
    kept_msg = f", {n_cells_kept} kept after cell_line filter" if cell_line_filter else ""
    print(f"==> walk done: {n_cells} cells{kept_msg}, {len(pert_sum)} unique perts, {walk_elapsed:.1f}s")

    # 3) Build per-pert × per-feature mean matrix
    perts = sorted(p for p in pert_sum if p in gains.index)
    print(f"    {len(perts)} perts overlap with eval set (out of {len(pert_sum)} walked, {len(gains)} in eval)")
    pert_mean = np.stack([pert_sum[p] / max(pert_count[p], 1) for p in perts], axis=0)  # (P, F)
    gain_arr = gains.loc[perts, gain_field].to_numpy()  # (P,)
    in_top_K = np.array([p in top_K_perts for p in perts])  # (P,)

    print(f"    perts in top-K and walked: {in_top_K.sum()} / {args.top_k_perts}")

    # 4) Per-feature scoring
    print("==> scoring features...")
    n_perts = pert_mean.shape[0]

    # Pearson r across all perts (vectorized)
    x = pert_mean - pert_mean.mean(axis=0, keepdims=True)
    y = gain_arr - gain_arr.mean()
    num = (x * y[:, None]).sum(axis=0)
    den = np.sqrt((x * x).sum(axis=0) * (y * y).sum())
    pearson_r = np.where(den > 0, num / den, np.nan)

    # Two-sided Pearson p-value via t-distribution
    with np.errstate(invalid="ignore", divide="ignore"):
        t_stat = pearson_r * np.sqrt(max(n_perts - 2, 1) / np.maximum(1 - pearson_r * pearson_r, 1e-12))
    pearson_p = 2 * scistats.t.sf(np.abs(t_stat), df=max(n_perts - 2, 1))

    # Top-K vs elsewhere: mean activation in top-K perts minus mean elsewhere
    if in_top_K.sum() > 0 and (~in_top_K).sum() > 0:
        mean_top_K = pert_mean[in_top_K].mean(axis=0)
        mean_other = pert_mean[~in_top_K].mean(axis=0)
        overrep = mean_top_K - mean_other
    else:
        mean_top_K = np.full(F, np.nan)
        mean_other = np.full(F, np.nan)
        overrep = np.full(F, np.nan)

    # 5) Join with index parquet (Δnorm, frac_active, etc.) for cross-reference
    index_path = out_dir / "index.parquet"
    if index_path.exists():
        idx = pd.read_parquet(index_path)
        print(f"    cross-referencing index.parquet ({len(idx)} rows)")
    else:
        idx = pd.DataFrame({"feature_id": np.arange(F, dtype=np.int64)})
        print(f"    (no index.parquet at {index_path}; cross-reference columns will be missing)")

    out = pd.DataFrame({
        "feature_id": np.arange(F, dtype=np.int64),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "mean_in_top_K_gain_perts": mean_top_K,
        "mean_elsewhere": mean_other,
        "overrepresentation": overrep,
        "n_perts_used": np.full(F, n_perts, dtype=np.int64),
        "top_K_perts": np.full(F, args.top_k_perts, dtype=np.int64),
    }).merge(idx, on="feature_id", how="left")

    out = out.sort_values("overrepresentation", ascending=False).reset_index(drop=True)
    suffix = f"_{cell_line_filter}" if cell_line_filter else ""
    out_path = out_dir / f"gain_attribution{suffix}.csv"
    out.to_csv(out_path, index=False)
    print(f"    wrote {out_path}  ({len(out)} rows)")

    # Summary callouts
    print("\n=== TOP 20 features by overrepresentation in high-gain perts ===")
    cols = ["feature_id", "overrepresentation", "pearson_r", "mean_in_top_K_gain_perts", "mean_elsewhere", "dec_norm_diff_b", "frac_active"]
    cols = [c for c in cols if c in out.columns]
    print(out.head(20)[cols].to_string(index=False))

    print("\n=== TOP 20 features by Pearson r ===")
    print(out.nlargest(20, "pearson_r")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
