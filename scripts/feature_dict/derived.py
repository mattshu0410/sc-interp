"""Single-card derived analyses.

These functions consume a feature card dict (from
:meth:`FeatureCard.to_json_dict`) and emit scalars/labels for paper
tables. No extra data sources required — see
:mod:`scripts.feature_dict.gain_correlation` for analyses that also need
eval-time per-pert gains.
"""
from __future__ import annotations


def active_cell_lines(
    card: dict,
    *,
    min_frac_active: float = 0.01,
    min_mean: float = 1e-3,
) -> list[str]:
    """Cell lines where the feature is non-trivially active.

    A line counts if at least ``min_frac_active`` of its tokens fired
    AND mean f exceeds ``min_mean``. Both thresholds are arguments
    because the right cutoff depends on dataset sparsity and feature
    magnitude.
    """
    out: list[str] = []
    for line, stats in card.get("by_cell_line", {}).items():
        if (
            stats.get("frac_active", 0.0) >= min_frac_active
            and stats.get("mean", 0.0) >= min_mean
        ):
            out.append(line)
    return sorted(out)


def best_concept_per_annotator(
    card: dict,
    *,
    min_neglog10_p: float = 0.0,
) -> dict[str, dict]:
    """Top hit per annotator that meets a significance floor.

    Returns ``{annotator_name: {concept_id, concept_name, score, ...}}``
    for annotators with at least one hit above ``min_neglog10_p``.
    """
    out: dict[str, dict] = {}
    for ann_name, hits in card.get("concepts", {}).items():
        if not hits:
            continue
        top = max(hits, key=lambda h: h["score"])
        if top["score"] < min_neglog10_p:
            continue
        out[ann_name] = {
            "concept_id": top["concept_id"],
            "concept_name": top["concept_name"],
            "score": top["score"],
            "score_kind": top["score_kind"],
            "n_overlap": top["n_genes_overlap"],
            "n_in_concept": top["n_genes_in_concept"],
        }
    return out


def transferable_concept_score(
    card: dict,
    *,
    min_cell_lines: int = 2,
    min_neglog10_p: float = 3.0,
) -> dict:
    """Flag features active in ≥k cell lines AND with a significant concept hit.

    The intuition: a feature firing only in one cell line is
    cell-line-specific. A feature active across cell lines whose top
    genes enrich for some biological concept is a candidate
    "transferable feature" — what we'd expect ESM to introduce if its
    prior teaches generalizable biology.

    `min_neglog10_p=3` corresponds to p < 1e-3 (stringent — picks only
    features with confident concept hits).
    """
    lines = active_cell_lines(card)
    concepts = best_concept_per_annotator(card, min_neglog10_p=min_neglog10_p)
    return {
        "transferable": len(lines) >= min_cell_lines and len(concepts) > 0,
        "active_cell_lines": lines,
        "n_active_cell_lines": len(lines),
        "best_concepts": concepts,
    }
