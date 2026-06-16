"""Render a per-feature summary CSV from a finished card directory.

Reads ``predictions/feature_dict/<tag>/cards/feature_*.json``, joins
per-pert gain from explicit eval-result paths, and produces
``summary.csv`` — one row per tracked feature.

Usage:

    python -m scripts.feature_dict.cli.summary \\
        --tag layer_7_topk_x8_k32_lr1e-04_steps50000 \\
        --primary-csv predictions/scgpt-replogle-esm-ft/eval/results.csv \\
        --baseline-csv base=predictions/scgpt-replogle-base-ft/eval/results.csv \\
        --baseline-csv random=predictions/scgpt-replogle-random-ft/eval/results.csv \\
        --gain-against base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.feature_dict.derived import (
    active_cell_lines,
    best_concept_per_annotator,
    transferable_concept_score,
)
from scripts.feature_dict.eval_join import join_eval_gains
from scripts.feature_dict.gain_correlation import gain_correlation


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "predictions" / "feature_dict"


def _parse_baseline(arg: str) -> tuple[str, Path]:
    if "=" not in arg:
        raise argparse.ArgumentTypeError(
            f"--baseline-csv expects LABEL=PATH, got {arg!r}"
        )
    label, path = arg.split("=", 1)
    return label, Path(path)


def _summarize(card: dict, gains: pd.DataFrame, gain_field: str) -> dict:
    lines = active_cell_lines(card)
    concepts = best_concept_per_annotator(card, min_neglog10_p=2.0)
    transfer = transferable_concept_score(card)
    gc = gain_correlation(card, gains, gain_field=gain_field)

    return {
        "feature_id": card["feature_id"],
        "dec_norm_diff_b": card["scores"]["dec_norm_diff_b"],
        "feature_norm_b": card["scores"]["feature_norm_b"],
        "frac_active": card["activation"]["frac_active"],
        "mean_when_active": card["activation"]["mean_when_active"],
        "max_activation": card["activation"]["max"],
        "top_5_genes": ", ".join(g["gene_symbol"] for g in card["top_genes"][:5]),
        "active_cell_lines": ",".join(lines),
        "n_active_cell_lines": len(lines),
        "transferable": transfer["transferable"],
        "best_hallmark": _fmt_concept(concepts.get("hallmark")),
        "best_reactome": _fmt_concept(concepts.get("reactome")),
        "best_go_bp": _fmt_concept(concepts.get("go_bp")),
        "best_go_mf": _fmt_concept(concepts.get("go_mf")),
        "best_go_cc": _fmt_concept(concepts.get("go_cc")),
        "best_kegg": _fmt_concept(concepts.get("kegg")),
        "best_wikipathways": _fmt_concept(concepts.get("wikipathways")),
        "best_progeny": _fmt_concept(concepts.get("progeny")),
        "best_collectri": _fmt_concept(concepts.get("collectri")),
        "gain_pearson_r": gc["pearson_r"],
        "gain_pearson_p": gc["pearson_p"],
        "gain_spearman_r": gc["spearman_r"],
        "gain_n_perts": gc["n_perts"],
        "gain_n_active_perts": gc["n_active_perts"],
        "top_perts_by_activation_x_gain": "; ".join(
            f"{r['pert']}(act={r['activation']:.4f},gain={r['gain']:+.3f})"
            for r in gc["top_perts_by_activation_x_gain"][:3]
        ),
    }


def _fmt_concept(c: dict | None) -> str:
    if c is None:
        return ""
    return f"{c['concept_name']} ({c['concept_id']}, -log10p={c['score']:.1f}, {c['n_overlap']}/{c['n_in_concept']})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, required=True,
                   help="Crosscoder tag (output subdir name under predictions/feature_dict/)")
    p.add_argument("--primary-csv", type=Path, required=True,
                   help="eval CSV for the 'clean' / primary condition (e.g. ESM)")
    p.add_argument("--baseline-csv", type=_parse_baseline, action="append", required=True,
                   metavar="LABEL=PATH",
                   help="eval CSV for a baseline condition. Repeatable.")
    p.add_argument("--gain-against", type=str, default=None,
                   help="Baseline label to correlate against (default: first baseline given).")
    p.add_argument("--metric", type=str, default="pearson_delta")
    p.add_argument("--primary-label", type=str, default="primary")
    args = p.parse_args()

    tag_dir = OUT_ROOT / args.tag
    cards_dir = tag_dir / "cards"
    if not cards_dir.exists():
        raise FileNotFoundError(f"missing cards dir at {cards_dir}; run build first")

    print(f"==> loading cards from {cards_dir}")
    cards: list[dict] = []
    for path in sorted(cards_dir.glob("feature_*.json")):
        with open(path) as f:
            cards.append(json.load(f))
    print(f"    {len(cards)} cards")

    print("==> joining eval gains")
    baselines = dict(args.baseline_csv)
    gains = join_eval_gains(
        primary_csv=args.primary_csv,
        primary_label=args.primary_label,
        baseline_csvs=baselines,
        metric=args.metric,
    )
    gain_field = args.gain_against or next(iter(baselines))
    gain_field_full = f"gain_vs_{gain_field}"
    if gain_field_full not in gains.columns:
        raise KeyError(
            f"--gain-against={gain_field!r} not in baselines {list(baselines)}; "
            f"available gain columns: {[c for c in gains.columns if c.startswith('gain_')]}"
        )
    print(f"    {len(gains)} perts; correlating against {gain_field_full!r} "
          f"(mean = {gains[gain_field_full].mean():+.3f})")

    print("==> summarizing")
    rows = [_summarize(c, gains, gain_field=gain_field_full) for c in cards]
    df = pd.DataFrame(rows)
    df = df.sort_values("dec_norm_diff_b", ascending=False).reset_index(drop=True)

    out = tag_dir / "summary.csv"
    df.to_csv(out, index=False)
    print(f"    wrote {out} ({len(df)} rows × {len(df.columns)} cols)")

    # Top finding callouts.
    print("\n=== TOP FEATURES BY GAIN-CORRELATION ===")
    top_corr = df.dropna(subset=["gain_pearson_r"]).nlargest(10, "gain_pearson_r")
    for _, r in top_corr.iterrows():
        print(
            f"  feat {int(r['feature_id']):>4d}: r={r['gain_pearson_r']:+.3f} (p={r['gain_pearson_p']:.1e})  "
            f"Δnorm_b={r['dec_norm_diff_b']:.3f}  lines={r['active_cell_lines']}  "
            f"top genes: {r['top_5_genes']}"
        )

    n_transferable = int(df["transferable"].sum())
    print(f"\n=== {n_transferable}/{len(df)} features flagged 'transferable' "
          f"(active in ≥2 cell lines AND have a -log10(p) ≥ 3 concept hit)")


if __name__ == "__main__":
    main()
