"""Re-annotate existing feature cards with decoupler annotators.

Loads the top_genes from each existing card and runs them through the
decoupler-based ConceptAnnotators. Writes a markdown + CSV table with the
top hit per ontology per feature. Skips ``go_bp`` (slow); keep using the
gProfiler-derived GO:BP from the original cards if you need it.

Usage:
    python -m scripts.feature_dict.cli.reannotate_decoupler \\
        --tag layer_7_topk_x8_k32_lr1e-04_steps50000 \\
        --top-features 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.feature_dict.annotators.decoupler_ import (
    DecouplerAnnotator,
    _collectri_loader,
    _hallmark_loader,
    _msigdb_collection_loader,
    _progeny_loader,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "predictions" / "feature_dict"


def _fast_annotators(cache_dir: Path) -> list[DecouplerAnnotator]:
    """All decoupler annotators except go_bp (which is ~7 min/feature)."""
    return [
        DecouplerAnnotator("hallmark",     _hallmark_loader(),               cache_dir),
        DecouplerAnnotator("reactome",     _msigdb_collection_loader("REACTOME_"), cache_dir),
        DecouplerAnnotator("go_mf",        _msigdb_collection_loader("GOMF_"),     cache_dir),
        DecouplerAnnotator("go_cc",        _msigdb_collection_loader("GOCC_"),     cache_dir),
        DecouplerAnnotator("kegg",         _msigdb_collection_loader("KEGG_"),     cache_dir),
        DecouplerAnnotator("wikipathways", _msigdb_collection_loader("WP_"),       cache_dir),
        DecouplerAnnotator("progeny",      _progeny_loader(),                cache_dir),
        DecouplerAnnotator("collectri",    _collectri_loader(),              cache_dir),
    ]


def _fmt_concept(h: dict | None) -> str:
    if h is None:
        return ""
    name = h["concept_name"][:55]
    return f"{name} (-log10p={h['score']:.1f}, {h['n_genes_overlap']}/{h['n_genes_in_concept']})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--top-features", type=int, default=10,
                   help="Number of top-Δnorm_b features to re-annotate (default: 10).")
    p.add_argument("--top-genes-per-feature", type=int, default=50)
    args = p.parse_args()

    tag_dir = OUT_ROOT / args.tag
    cards_dir = tag_dir / "cards"
    cache_dir = OUT_ROOT / "_annotator_cache_decoupler"

    cards = sorted(cards_dir.glob("feature_*.json"))
    if not cards:
        raise FileNotFoundError(f"no cards in {cards_dir}")
    print(f"==> loaded {len(cards)} cards from {cards_dir}")

    # Sort cards by Δnorm_b descending; take top-K.
    rows = [json.loads(p.read_text()) for p in cards]
    rows.sort(key=lambda c: -c["scores"]["dec_norm_diff_b"])
    rows = rows[: args.top_features]
    print(f"    re-annotating top {len(rows)} by Δnorm_b "
          f"(range {rows[-1]['scores']['dec_norm_diff_b']:.3f}–{rows[0]['scores']['dec_norm_diff_b']:.3f})")

    annotators = _fast_annotators(cache_dir)
    print(f"==> running {len(annotators)} annotators per feature "
          f"(go_bp excluded; use existing card's gprofiler:GO:BP if needed)")

    table_rows: list[dict] = []
    for fi, card in enumerate(rows):
        feat = card["feature_id"]
        dnd = card["scores"]["dec_norm_diff_b"]
        top_genes = card["top_genes"][: args.top_genes_per_feature]
        gene_weights = {
            g["gene_symbol"]: float(g.get("mean_when_present", 1.0))
            for g in top_genes
        }
        top5_str = ", ".join(g["gene_symbol"] for g in top_genes[:5])
        print(f"\n[{fi+1}/{len(rows)}] feature {feat}  Δnorm_b={dnd:.3f}  top: {top5_str}")

        out: dict = {
            "feature_id": feat,
            "dec_norm_diff_b": dnd,
            "top_5_genes": top5_str,
            "frac_active": card["activation"]["frac_active"],
        }
        for ann in annotators:
            t0 = time.time()
            try:
                hits = ann.annotate(gene_weights, background=None, method="hypergeom")
            except Exception as e:
                print(f"    {ann.name}: {type(e).__name__}: {e}")
                hits = []
            elapsed = time.time() - t0
            top_hit = hits[0] if hits else None
            top_hit_d = (
                {
                    "concept_name": top_hit.concept_name,
                    "score": top_hit.score,
                    "n_genes_overlap": top_hit.n_genes_overlap,
                    "n_genes_in_concept": top_hit.n_genes_in_concept,
                }
                if top_hit
                else None
            )
            out[f"best_{ann.name}"] = _fmt_concept(top_hit_d)
            print(f"    {ann.name:<14s}  {elapsed:>5.1f}s  top: {_fmt_concept(top_hit_d)[:60]}")
        table_rows.append(out)

    # Write outputs
    df = pd.DataFrame(table_rows)
    csv_path = tag_dir / f"reannotate_decoupler_top{args.top_features}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n==> wrote {csv_path}")

    # Markdown table for the paper
    md_path = tag_dir / f"reannotate_decoupler_top{args.top_features}.md"
    with md_path.open("w") as f:
        f.write(f"# Top-{args.top_features} ESM-specific crosscoder features (decoupler annotation)\n\n")
        f.write(f"Crosscoder: `{args.tag}`. Selection: top by `feature_dec_norm_diff_b`. "
                f"Annotated with decoupler against OmniPath/MSigDB resources.\n\n")
        cols = ["feature_id", "dec_norm_diff_b", "frac_active", "top_5_genes",
                "best_hallmark", "best_reactome", "best_go_mf", "best_go_cc",
                "best_kegg", "best_wikipathways", "best_progeny", "best_collectri"]
        cols = [c for c in cols if c in df.columns]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v) if v else "")
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"==> wrote {md_path}")


if __name__ == "__main__":
    main()
