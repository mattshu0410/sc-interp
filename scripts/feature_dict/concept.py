"""ConceptAnnotator Protocol and `ConceptHit` dataclass.

An annotator maps a gene set (or weighted gene-activation profile) to a
ranked list of concept hits against some ontology. Implementations are
in :mod:`scripts.feature_dict.annotators`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol


ScoreKind = Literal["neglog10_p", "neglog10_q", "auroc", "log_odds", "nes"]


@dataclass(frozen=True)
class ConceptHit:
    """One ontology hit with overlap statistics."""
    annotator: str                  # e.g. "gprofiler:GO:BP"
    concept_id: str                 # ontology-native id (e.g. "GO:0006259")
    concept_name: str
    score: float                    # interpret per `score_kind`
    score_kind: ScoreKind
    n_genes_in_concept: int         # term size
    n_genes_query: int              # query size
    n_genes_overlap: int            # intersection size
    context: dict = field(default_factory=dict)


class ConceptAnnotator(Protocol):
    """Maps a gene set to ranked ConceptHits.

    All annotators share this interface. The walker / card builder calls
    `annotate` and never branches on annotator type.
    """

    name: str               # display name
    score_kind: ScoreKind   # for `ConceptHit.score`

    def annotate(
        self,
        gene_weights: dict[str, float],
        background: set[str] | None = None,
        method: Literal["hypergeom", "gsea"] = "hypergeom",
    ) -> list[ConceptHit]: ...


# ── On-disk cache ───────────────────────────────────────────────────────────


def gene_set_hash(
    annotator_key: str,
    gene_weights: dict[str, float],
    background: set[str] | None,
    method: str,
    organism: str = "hsapiens",
) -> str:
    """Stable sha1 over the inputs that determine an annotation result.

    Gene weights are sorted by symbol; weights are quantized to 6 decimal
    places to avoid float-jitter cache misses.
    """
    items = sorted(gene_weights.items())
    payload = {
        "annotator": annotator_key,
        "method": method,
        "organism": organism,
        "genes": [(k, round(v, 6)) for k, v in items],
        "background_n": len(background) if background else 0,
        "background_hash": (
            hashlib.sha1(",".join(sorted(background)).encode()).hexdigest()[:16]
            if background else "none"
        ),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()


def cache_load(cache_dir: Path, key: str) -> list[ConceptHit] | None:
    p = cache_dir / f"{key}.json"
    if not p.exists():
        return None
    with open(p) as f:
        rows = json.load(f)
    return [ConceptHit(**r) for r in rows]


def cache_store(cache_dir: Path, key: str, hits: list[ConceptHit]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{key}.json"
    with open(p, "w") as f:
        json.dump([asdict(h) for h in hits], f, indent=2)
