"""Registry of activation captures by friendly condition name.

Each condition points at a directory holding `shard-*.h5` + `stats.h5`,
matching the H5ActivationSink folder layout. Repeval scripts resolve
condition names to paths via :func:`resolve` so analyses can reference
``base`` / ``esm`` / ``random`` without hard-coding paths.

To add a condition: drop a capture folder into
``predictions/scgpt-replogle-activations/<name>/`` and add an entry
below.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACTS_ROOT = REPO_ROOT / "predictions" / "scgpt-replogle-activations"

CONDITIONS: dict[str, Path] = {
    "base":   ACTS_ROOT / "base",
    "esm":    ACTS_ROOT / "esm",
    "random": ACTS_ROOT / "random",
}


def resolve(name_or_path: str | Path) -> Path:
    """Return the capture folder for a registered name, or accept a raw path.

    Raises KeyError if the name is unknown and the path does not exist.
    """
    if isinstance(name_or_path, Path):
        return name_or_path
    if name_or_path in CONDITIONS:
        return CONDITIONS[name_or_path]
    p = Path(name_or_path)
    if p.exists():
        return p
    raise KeyError(
        f"unknown condition {name_or_path!r}; "
        f"registered: {sorted(CONDITIONS)}; or pass an existing path"
    )
