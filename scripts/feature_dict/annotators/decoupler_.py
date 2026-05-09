"""decoupler-backed ConceptAnnotator using OmniPath gene-set resources.

Uses ``decoupler.mt.query_set`` (Fisher's exact / hypergeometric) against
networks loaded via ``decoupler.op``. One annotator instance per resource;
factory helpers cover the common biological databases.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd


@functools.lru_cache(maxsize=4)
def _load_msigdb(organism: str = "human") -> pd.DataFrame:
    """Cached MSigDB resource (5.9M rows). Avoids repeated downloads when
    multiple collection-filtered annotators are constructed."""
    import decoupler as dc
    return dc.op.resource("MSigDB", organism=organism)

from scripts.feature_dict.concept import (
    ConceptHit,
    ScoreKind,
    cache_load,
    cache_store,
    gene_set_hash,
)


@dataclass
class DecouplerAnnotator:
    """ConceptAnnotator backed by ``decoupler.mt.query_set``.

    Args:
        name: Display name (e.g. ``"hallmark"``, ``"reactome"``,
            ``"go_bp"``). Used as the annotator key in cache + cards.
        net_loader: Callable that returns a DataFrame with ``source`` and
            ``target`` columns (gene-set name → gene symbol). Lazy:
            called once per process.
        cache_dir: per-gene-set on-disk cache root.
        organism: passed to OmniPath loaders.
        tmin: minimum genes per source (sources with fewer are dropped).
        score_kind: how to interpret ``ConceptHit.score``.
    """
    name: str
    net_loader: Callable[[], pd.DataFrame]
    cache_dir: Path
    organism: str = "human"
    tmin: int = 5
    score_kind: ScoreKind = "neglog10_p"

    @cached_property
    def _net(self) -> pd.DataFrame:
        df = self.net_loader()
        if "source" not in df.columns or "target" not in df.columns:
            raise ValueError(
                f"{self.name}: net_loader must return DataFrame with "
                f"'source' and 'target' columns; got {list(df.columns)}"
            )
        return df

    @cached_property
    def _source_sizes(self) -> pd.Series:
        """Pre-computed per-source target count, used to populate
        ConceptHit.n_genes_in_concept without rebuilding sets per query."""
        return self._net.groupby("source")["target"].nunique()

    def _overlap_counts(self, query: set[str]) -> pd.Series:
        """Per-source count of targets that are in the query set. O(net_rows)."""
        if not query:
            return pd.Series(dtype=np.int64)
        sub = self._net[self._net["target"].isin(query)]
        return sub.groupby("source")["target"].nunique()

    def annotate(
        self,
        gene_weights: dict[str, float],
        background: set[str] | None = None,
        method: Literal["hypergeom", "gsea"] = "hypergeom",
    ) -> list[ConceptHit]:
        if method != "hypergeom":
            raise NotImplementedError(
                f"{self.name}: only hypergeom (decoupler.mt.query_set) is "
                "wired up. Use a separate annotator for GSEA preranked."
            )

        key = gene_set_hash(self.name, gene_weights, background, method, self.organism)
        cached = cache_load(self.cache_dir, key)
        if cached is not None:
            return cached

        features = list(gene_weights.keys())
        if not features:
            return []

        # decoupler imported lazily so the module can be parsed even when
        # the venv lacks decoupler.
        import decoupler as dc

        n_bg = len(background) if background else 20000
        try:
            res = dc.mt.query_set(
                features=features,
                net=self._net,
                alternative="greater",
                n_bg=n_bg,
                tmin=self.tmin,
                verbose=False,
            )
        except Exception as e:
            print(f"  {self.name}: dc.mt.query_set raised {type(e).__name__}: {e}")
            return []

        if res is None or len(res) == 0:
            return []

        # query_set returns: source, stat (odds ratio), pval, padj.
        # Drop sources with zero overlap upfront — they can't be enriched and
        # we don't want to write thousands of zero-hit rows to the on-disk cache.
        feature_set = set(features)
        overlaps = self._overlap_counts(feature_set)            # O(net_rows)
        sizes = self._source_sizes                               # cached
        # Sort by p-value ascending (most significant first) and only keep
        # those with at least one overlapping gene.
        res = res.sort_values("pval")
        hits: list[ConceptHit] = []
        for _, row in res.iterrows():
            src = str(row["source"])
            ov = int(overlaps.get(src, 0))
            if ov == 0:
                continue
            pval = float(max(row.get("pval", 1.0), 1e-300))
            padj = float(row.get("padj", pval))
            hits.append(ConceptHit(
                annotator=self.name,
                concept_id=src,
                concept_name=src,
                score=float(-np.log10(pval)),
                score_kind=self.score_kind,
                n_genes_in_concept=int(sizes.get(src, 0)),
                n_genes_query=len(features),
                n_genes_overlap=ov,
                context={
                    "p_value": pval,
                    "padj": padj,
                    "odds_ratio": float(row.get("stat", 0.0)),
                },
            ))
        cache_store(self.cache_dir, key, hits)
        return hits


# ── Network loaders ─────────────────────────────────────────────────────────


def _hallmark_loader(organism: str = "human") -> Callable[[], pd.DataFrame]:
    def load() -> pd.DataFrame:
        import decoupler as dc
        return dc.op.hallmark(organism=organism)
    return load


def _progeny_loader(
    organism: str = "human", *, top: int = 500
) -> Callable[[], pd.DataFrame]:
    """PROGENy with a per-pathway gene cap. Default top=500 follows
    common practice (raw PROGENy can include thousands of weighted
    targets per pathway, which dilutes ORA)."""
    def load() -> pd.DataFrame:
        import decoupler as dc
        return dc.op.progeny(organism=organism, top=top)
    return load


def _collectri_loader(organism: str = "human") -> Callable[[], pd.DataFrame]:
    def load() -> pd.DataFrame:
        import decoupler as dc
        return dc.op.collectri(organism=organism)
    return load


def _msigdb_collection_loader(
    prefix: str, organism: str = "human"
) -> Callable[[], pd.DataFrame]:
    """Filter MSigDB by geneset-name prefix → (source, target) net.

    MSigDB collections share a prefix in the geneset name:
      * ``REACTOME_`` → Reactome
      * ``GOBP_`` / ``GOMF_`` / ``GOCC_`` → GO Biological Process / Mol Function / Cell Component
      * ``KEGG_LEGACY_`` / ``KEGG_MEDICUS_`` → KEGG
      * ``WP_`` → WikiPathways
      * ``HALLMARK_`` → MSigDB Hallmarks (also available via dc.op.hallmark)
    """
    def load() -> pd.DataFrame:
        msigdb = _load_msigdb(organism)
        sub = msigdb[msigdb["geneset"].str.startswith(prefix)].copy()
        sub = sub.rename(columns={"genesymbol": "target", "geneset": "source"})
        # MSigDB has duplicate (source, target) rows from overlapping collections
        # (e.g. a gene appearing in multiple Hallmark sets). decoupler.mt.query_set
        # asserts uniqueness; drop dupes.
        return sub[["source", "target"]].drop_duplicates().reset_index(drop=True)
    return load
