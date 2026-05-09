"""End-to-end driver: walk pair → aggregate → annotate → write cards + index.

Run via:

    source tools/diffing/.venv/bin/activate
    python -m scripts.feature_dict.cli.build \\
        --model-dir predictions/diff/scgpt_base_vs_esm_replogle/crosscoder/layer_7/topk_x8_k32_lr1e-04_steps50000 \\
        --capture transformer_encoder.layers.7 \\
        --base predictions/scgpt-replogle-activations/base \\
        --esm  predictions/scgpt-replogle-activations/esm \\
        --top-k 50

Writes:
    predictions/feature_dict/<tag>/index.parquet
    predictions/feature_dict/<tag>/cards/feature_<id>.json
    predictions/feature_dict/_annotator_cache/<sha1>.json   (shared)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.diffing.pairs import load_pair
from scripts.feature_dict.annotators import default_annotators
from scripts.feature_dict.card import FeatureCard
from scripts.feature_dict.sources.crosscoder import CrosscoderSource
from scripts.feature_dict.stats import Aggregator
from scripts.feature_dict.walker import resolve_gene_symbols, walk_pair


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = REPO_ROOT / "predictions" / "feature_dict"
CACHE_DIR = OUT_ROOT / "_annotator_cache"


def _crosscoder_tag(model_dir: Path) -> str:
    """Stable tag from the model dir, e.g. ``layer_7_topk_x8_k32...``.

    Uses the last two segments of the model dir path (layer + config_tag).
    """
    parts = model_dir.resolve().parts
    return "_".join(parts[-2:])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--capture", type=str, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--esm", type=Path, required=True)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--chunk-rows", type=int, default=16)
    p.add_argument("--top-genes-for-enrichment", type=int, default=50)
    p.add_argument("--phase", type=str, default="predict")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-annotate", action="store_true",
                   help="Skip ontology annotation; write index + cards with empty concept lists.")
    p.add_argument("--limit-cells", type=int, default=None,
                   help="Cap total cells walked (for quick dry runs).")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tags = {"phase": args.phase}
    tag = _crosscoder_tag(args.model_dir)
    out_dir = OUT_ROOT / tag
    cards_dir = out_dir / "cards"
    out_dir.mkdir(parents=True, exist_ok=True)
    cards_dir.mkdir(parents=True, exist_ok=True)

    print(f"==> tag: {tag}")
    print(f"    output: {out_dir}")

    # 1) Load pair, source, gene vocab
    pair = load_pair(args.base, args.esm, capture=args.capture, tags_a=tags, tags_b=tags)
    src = CrosscoderSource(model_dir=args.model_dir, device=device)
    symbols = resolve_gene_symbols(args.esm)
    print(f"    dict_size={src.dict_size}  gene_vocab={len(symbols)}")

    # 2) Pick the top-K ESM-specific features by Δnorm_b (Minder 2025 §3.1)
    tracked = src.select_features(args.top_k)
    print(f"    tracking top-{args.top_k} features by feature_dec_norm_diff_b")
    print(f"    Δnorm_b range tracked: "
          f"[{src.scores['dec_norm_diff_b'][tracked].min():.3f}, "
          f"{src.scores['dec_norm_diff_b'][tracked].max():.3f}]")

    # 3) Walk + aggregate
    agg = Aggregator(n_features=src.dict_size, tracked_features=tracked, n_genes=len(symbols))
    print("==> aggregating...")
    t0 = time.time()
    n_cells = 0
    for i, chunk in enumerate(walk_pair(pair, src, chunk_rows=args.chunk_rows, device=device)):
        agg.update(chunk)
        n_cells += chunk.codes.shape[0]
        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            print(f"    chunk {i+1} | {n_cells} cells | {elapsed:.1f}s ({n_cells/elapsed:.1f} cells/s)")
        if args.limit_cells is not None and n_cells >= args.limit_cells:
            print(f"    --limit-cells={args.limit_cells} hit; stopping walk")
            break
    print(f"==> walk complete: {n_cells} cells in {time.time()-t0:.1f}s")

    index_stats = agg.finalize_index()
    rich = agg.finalize_cards(symbols, top_k_genes=args.top_genes_for_enrichment)

    # 4) Annotate
    annotators = default_annotators(CACHE_DIR) if not args.no_annotate else []
    if annotators:
        print(f"==> running {len(annotators)} annotators × {args.top_k} features (cached by gene-set hash)")
    background = set(symbols.tolist())
    feature_concepts: dict[int, dict[str, list]] = {}
    t1 = time.time()
    for fi, feat in enumerate(tracked):
        gene_set = {
            g["gene_symbol"]: float(g["mean_when_present"])
            for g in rich[feat]["top_genes"]
            if g["mean_when_present"] > 0
        }
        feature_concepts[feat] = {}
        if not annotators or not gene_set:
            continue
        for ann in annotators:
            try:
                hits = ann.annotate(gene_set, background=background, method="hypergeom")
            except Exception as e:
                print(f"    feat {feat} annotator {ann.name}: {type(e).__name__}: {e}")
                hits = []
            feature_concepts[feat][ann.name] = hits
        if (fi + 1) % 10 == 0:
            print(f"    annotated {fi+1}/{len(tracked)} features ({time.time()-t1:.1f}s)")
    if annotators:
        print(f"==> annotation complete: {time.time()-t1:.1f}s")

    # 5) Write deep cards
    print("==> writing cards...")
    for feat in tracked:
        card = FeatureCard(
            feature_id=int(feat),
            crosscoder_tag=tag,
            scores={
                "dec_norm_diff_a": float(src.scores["dec_norm_diff_a"][feat]),
                "dec_norm_diff_b": float(src.scores["dec_norm_diff_b"][feat]),
                "feature_norm_a": float(src.scores["feature_norm_a"][feat]),
                "feature_norm_b": float(src.scores["feature_norm_b"][feat]),
            },
            activation={
                "mean": float(index_stats["mean"][feat]),
                "std": float(index_stats["std"][feat]),
                "max": float(index_stats["max"][feat]),
                "frac_active": float(index_stats["frac_active"][feat]),
                "mean_when_active": float(index_stats["mean_when_active"][feat]),
                "active_count": int(index_stats["active_count"][feat]),
                "total_count": int(index_stats["total_count"]),
            },
            top_genes=rich[feat]["top_genes"],
            top_cells=rich[feat]["top_cells"],
            by_cell_line=rich[feat]["by_cell_line"],
            by_pert=rich[feat]["by_pert"],
            concepts=feature_concepts[feat],
        )
        card.write(cards_dir)
    print(f"    {len(tracked)} cards in {cards_dir}")

    # 6) Write index parquet (all features × cheap stats)
    print("==> writing index parquet...")
    F = src.dict_size
    df = pd.DataFrame({
        "feature_id": np.arange(F, dtype=np.int64),
        "feature_norm_a": src.scores["feature_norm_a"],
        "feature_norm_b": src.scores["feature_norm_b"],
        "dec_norm_diff_a": src.scores["dec_norm_diff_a"],
        "dec_norm_diff_b": src.scores["dec_norm_diff_b"],
        "mean": index_stats["mean"],
        "std": index_stats["std"],
        "max": index_stats["max"],
        "frac_active": index_stats["frac_active"],
        "mean_when_active": index_stats["mean_when_active"],
        "active_count": index_stats["active_count"],
    })
    df["is_tracked"] = df["feature_id"].isin(tracked)
    df["esm_specific_rank"] = df["dec_norm_diff_b"].rank(ascending=False, method="min").astype(np.int64)
    parquet_path = out_dir / "index.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"    {parquet_path} ({len(df)} rows)")

    print("\n==> done")


if __name__ == "__main__":
    main()
