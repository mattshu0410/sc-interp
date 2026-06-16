"""Walk a chosen set of feature IDs on rpe1 cells and annotate via decoupler.

Intended for symmetric exploration: top-K by Δnorm ascending = base-specific,
descending = ESM-specific. Lighter than build.py because it doesn't write
full cards — just enough to run the annotators.

Usage:
    python -m scripts.feature_dict.cli.annotate_features \\
        --tag layer_7_topk_x8_k32_lr1e-04_steps50000 \\
        --model-dir <crosscoder dir> \\
        --capture transformer_encoder.layers.7 \\
        --base predictions/scgpt-replogle-activations/base \\
        --esm  predictions/scgpt-replogle-activations/esm \\
        --select base   # or 'esm'
        --top-k 10
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.diffing.pairs import load_pair
from scripts.feature_dict.annotators.decoupler_ import (
    DecouplerAnnotator,
    _collectri_loader,
    _hallmark_loader,
    _msigdb_collection_loader,
    _progeny_loader,
)
from scripts.feature_dict.sources.crosscoder import CrosscoderSource
from scripts.feature_dict.stats import Aggregator
from scripts.feature_dict.walker import resolve_gene_symbols, walk_pair


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "predictions" / "feature_dict"


def _fast_annotators(cache_dir: Path) -> list[DecouplerAnnotator]:
    return [
        DecouplerAnnotator("hallmark",     _hallmark_loader(),                cache_dir),
        DecouplerAnnotator("reactome",     _msigdb_collection_loader("REACTOME_"), cache_dir),
        DecouplerAnnotator("go_mf",        _msigdb_collection_loader("GOMF_"),     cache_dir),
        DecouplerAnnotator("go_cc",        _msigdb_collection_loader("GOCC_"),     cache_dir),
        DecouplerAnnotator("kegg",         _msigdb_collection_loader("KEGG_"),     cache_dir),
        DecouplerAnnotator("wikipathways", _msigdb_collection_loader("WP_"),       cache_dir),
        DecouplerAnnotator("progeny",      _progeny_loader(),                 cache_dir),
        DecouplerAnnotator("collectri",    _collectri_loader(),               cache_dir),
    ]


def _fmt(h) -> str:
    if h is None:
        return ""
    return f"{h.concept_name[:55]} (-log10p={h.score:.1f}, {h.n_genes_overlap}/{h.n_genes_in_concept})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", type=str, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--capture", type=str, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--esm", type=Path, required=True)
    p.add_argument("--select", choices=["esm", "base"], default="base",
                   help="esm = top-K by largest Δnorm_b (ESM-specific); "
                        "base = top-K by smallest Δnorm_b (base-specific).")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--cell-line", type=str, default="rpe1")
    p.add_argument("--top-genes", type=int, default=50)
    p.add_argument("--phase", type=str, default="predict")
    p.add_argument("--chunk-rows", type=int, default=32)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tags = {"phase": args.phase}
    out_dir = OUT_ROOT / args.tag
    cache_dir = OUT_ROOT / "_annotator_cache_decoupler"

    pair = load_pair(args.base, args.esm, capture=args.capture, tags_a=tags, tags_b=tags)
    src = CrosscoderSource(model_dir=args.model_dir, device=device)
    symbols = resolve_gene_symbols(args.esm)

    dnd_b = src.scores["dec_norm_diff_b"]
    if args.select == "esm":
        feature_ids = np.argsort(-dnd_b)[:args.top_k].tolist()
        label = "ESM-specific (largest Δnorm_b)"
    else:
        feature_ids = np.argsort(dnd_b)[:args.top_k].tolist()
        label = "base-specific (smallest Δnorm_b)"
    print(f"==> selected top-{args.top_k} {label}: {feature_ids}")
    print(f"    Δnorm_b range: [{dnd_b[feature_ids].min():.4f}, {dnd_b[feature_ids].max():.4f}]")

    # Walk rpe1 cells, aggregate top-genes per tracked feature.
    print(f"==> walking {args.cell_line} cells...")
    agg = Aggregator(n_features=src.dict_size, tracked_features=feature_ids, n_genes=len(symbols))
    t0 = time.time()
    n_kept = 0
    for i, chunk in enumerate(walk_pair(pair, src, chunk_rows=args.chunk_rows, device=device)):
        if args.cell_line:
            keep = chunk.cell_line == args.cell_line
            if not keep.any():
                continue
            # Subset chunk to rpe1
            from scripts.feature_dict.walker import Chunk
            chunk = Chunk(
                codes=chunk.codes[torch.from_numpy(keep)],
                cell_id=chunk.cell_id[keep],
                pert=chunk.pert[keep],
                cell_line=chunk.cell_line[keep],
                gene_dataset_ids=chunk.gene_dataset_ids[keep],
            )
            n_kept += int(keep.sum())
        agg.update(chunk)
        if (i + 1) % 25 == 0:
            print(f"    chunk {i+1}, {n_kept} {args.cell_line} cells aggregated, {time.time()-t0:.1f}s")
    print(f"==> walk done: {n_kept} {args.cell_line} cells, {time.time()-t0:.1f}s")

    rich = agg.finalize_cards(symbols, top_k_genes=args.top_genes)

    # Annotate
    annotators = _fast_annotators(cache_dir)
    table_rows: list[dict] = []
    for fi, feat in enumerate(feature_ids):
        rs = rich[feat]
        gene_weights = {
            g["gene_symbol"]: float(g.get("mean_when_present", 1.0))
            for g in rs["top_genes"]
            if g.get("mean_when_present", 0.0) > 0
        }
        top5 = ", ".join(g["gene_symbol"] for g in rs["top_genes"][:5])
        print(f"\n[{fi+1}/{len(feature_ids)}] feature {feat}  Δnorm_b={dnd_b[feat]:.3f}  top: {top5}")
        if not gene_weights:
            print("    (no active gene tokens; skipping annotation)")
            continue
        out = {
            "feature_id": int(feat),
            "dec_norm_diff_b": float(dnd_b[feat]),
            "feature_norm_a": float(src.scores["feature_norm_a"][feat]),
            "feature_norm_b": float(src.scores["feature_norm_b"][feat]),
            "top_5_genes": top5,
        }
        for ann in annotators:
            t0 = time.time()
            try:
                hits = ann.annotate(gene_weights, background=None, method="hypergeom")
            except Exception as e:
                print(f"    {ann.name}: {type(e).__name__}: {e}")
                hits = []
            top_hit = hits[0] if hits else None
            out[f"best_{ann.name}"] = _fmt(top_hit)
            print(f"    {ann.name:<14s}  {time.time()-t0:>5.1f}s  top: {_fmt(top_hit)[:60]}")
        table_rows.append(out)

    df = pd.DataFrame(table_rows)
    suffix = f"_{args.select}_top{args.top_k}_{args.cell_line}"
    csv_path = out_dir / f"annotate{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n==> wrote {csv_path}")
    md_path = out_dir / f"annotate{suffix}.md"
    with md_path.open("w") as f:
        f.write(f"# Top-{args.top_k} {label} crosscoder features (decoupler annotation, {args.cell_line})\n\n")
        cols = ["feature_id", "dec_norm_diff_b", "feature_norm_a", "feature_norm_b", "top_5_genes",
                "best_hallmark", "best_reactome", "best_go_mf", "best_go_cc",
                "best_kegg", "best_wikipathways", "best_progeny", "best_collectri"]
        cols = [c for c in cols if c in df.columns]
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for _, row in df.iterrows():
            cells = [(f"{v:.3f}" if isinstance(v, float) else str(v) if v else "") for v in (row[c] for c in cols)]
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"==> wrote {md_path}")


if __name__ == "__main__":
    main()
