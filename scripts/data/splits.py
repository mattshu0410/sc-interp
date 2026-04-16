"""Canonical train/val/test split artifacts.

Splits live at data/<name>/splits/<split_type>_<seed>_<tgss>.json and
carry the perturbation-label sets for each partition. This module is
source-agnostic: it knows the JSON schema and the path convention, and
nothing about how the splits were produced. Runners import from here;
source-specific materialisers (scripts.data.gears, future scripts.data.*)
write files in this shape.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.manifest import REPO_ROOT, Manifest


def canonical_split_path(
    dataset_name: str,
    split_type: str,
    seed: int,
    train_gene_set_size: float,
    repo_root: Path = REPO_ROOT,
) -> Path:
    """Path where a canonical split JSON lives for the given parameters."""
    return (
        repo_root
        / "data"
        / dataset_name
        / "splits"
        / f"{split_type}_{seed}_{train_gene_set_size}.json"
    )


def load_split(
    manifest: Manifest,
    split_type: str,
    seed: int,
    train_gene_set_size: float,
) -> dict[str, Any]:
    """Read the canonical split JSON for this manifest + split parameters.

    Returns a dict with keys `train`, `val`, `test` (lists of perturbation
    labels) plus the split parameters themselves. Raises FileNotFoundError
    with an actionable message if the artifact is missing.
    """
    path = canonical_split_path(manifest.name, split_type, seed, train_gene_set_size)
    if not path.exists():
        raise FileNotFoundError(
            f"canonical split missing at {path}. "
            f"materialise it by running:\n"
            f"    source tools/.venv/bin/activate\n"
            f"    python -m scripts.data.{manifest.source} --dataset {manifest.name}"
        )
    with open(path) as f:
        return json.load(f)


def write_split(
    dataset_name: str,
    split_type: str,
    seed: int,
    train_gene_set_size: float,
    train: list[str],
    val: list[str],
    test: list[str],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a canonical split JSON and return the path it was written to."""
    path = canonical_split_path(dataset_name, split_type, seed, train_gene_set_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "split_type": split_type,
        "seed": seed,
        "train_gene_set_size": train_gene_set_size,
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
