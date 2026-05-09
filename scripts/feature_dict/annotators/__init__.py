"""Concept annotators implementing the ConceptAnnotator Protocol.

Default annotator set uses :class:`DecouplerAnnotator` against
OmniPath/MSigDB resources via the ``decoupler`` package:

* MSigDB Hallmark — broad transcriptional signatures (50 sets)
* Reactome pathways
* GO Biological Process / Molecular Function / Cellular Component
* KEGG pathways
* WikiPathways
* PROGENy — pathway-responsive gene sets
* CollecTRI — TF → target regulatory edges
"""
from __future__ import annotations

from pathlib import Path

from scripts.feature_dict.annotators.decoupler_ import (
    DecouplerAnnotator,
    _collectri_loader,
    _hallmark_loader,
    _msigdb_collection_loader,
    _progeny_loader,
)


def default_annotators(
    cache_dir: Path,
    *,
    organism: str = "human",
) -> list:
    """Workshop-default annotator set, all decoupler-backed.

    Caches results to ``cache_dir`` keyed by gene-set hash so overlapping
    queries across features share results.
    """
    return [
        DecouplerAnnotator(
            name="hallmark",
            net_loader=_hallmark_loader(organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="reactome",
            net_loader=_msigdb_collection_loader("REACTOME_", organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="go_bp",
            net_loader=_msigdb_collection_loader("GOBP_", organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="go_mf",
            net_loader=_msigdb_collection_loader("GOMF_", organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="go_cc",
            net_loader=_msigdb_collection_loader("GOCC_", organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="kegg",
            net_loader=_msigdb_collection_loader("KEGG_", organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="wikipathways",
            net_loader=_msigdb_collection_loader("WP_", organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="progeny",
            net_loader=_progeny_loader(organism),
            cache_dir=cache_dir, organism=organism,
        ),
        DecouplerAnnotator(
            name="collectri",
            net_loader=_collectri_loader(organism),
            cache_dir=cache_dir, organism=organism,
        ),
    ]
