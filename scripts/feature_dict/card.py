"""FeatureCard: composite per-feature record (stats + concepts)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scripts.feature_dict.concept import ConceptHit


@dataclass(frozen=True)
class FeatureCard:
    """One feature's complete annotation record.

    Inputs come from three sources:
    * ``scores`` — from the crosscoder's ``scores.h5`` (decoder geometry).
    * ``activation``, ``top_genes``, ``top_cells``, ``by_cell_line``,
      ``by_pert`` — from :class:`scripts.feature_dict.stats.RichStats`
      and :class:`scripts.feature_dict.stats.ScalarStats`.
    * ``concepts`` — per-annotator lists of :class:`ConceptHit` from
      :mod:`scripts.feature_dict.annotators`.
    """
    feature_id: int
    crosscoder_tag: str
    scores: dict[str, float]
    activation: dict[str, float]
    top_genes: list[dict]
    top_cells: list[dict]
    by_cell_line: dict[str, dict]
    by_pert: dict[str, dict]
    concepts: dict[str, list[ConceptHit]] = field(default_factory=dict)

    def to_json_dict(self) -> dict:
        return {
            "feature_id": self.feature_id,
            "crosscoder_tag": self.crosscoder_tag,
            "scores": self.scores,
            "activation": self.activation,
            "top_genes": self.top_genes,
            "top_cells": self.top_cells,
            "by_cell_line": self.by_cell_line,
            "by_pert": self.by_pert,
            "concepts": {
                k: [asdict(h) for h in v] for k, v in self.concepts.items()
            },
        }

    def write(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"feature_{self.feature_id:04d}.json"
        with open(p, "w") as f:
            json.dump(self.to_json_dict(), f, indent=2)
        return p
