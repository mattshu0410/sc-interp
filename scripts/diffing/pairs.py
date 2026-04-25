"""Build a DiffPair from h5 activation paths."""

from __future__ import annotations

from pathlib import Path

from scripts.diffing.alignment import Alignment, Identity, get_alignment
from scripts.diffing.base import ActivationSource, DiffPair, Relationship


def load_pair(
    path_a: str | Path,
    path_b: str | Path,
    capture: str,
    *,
    alignment: Alignment | str = "identity",
    relationship: Relationship = "unspecified",
    tags_a: dict[str, str] | None = None,
    tags_b: dict[str, str] | None = None,
) -> DiffPair:
    """Construct a DiffPair without loading any activations.

    The DiffPair row-count check runs at construction, which opens the h5
    metadata but not the dataset payload. Use `iter_chunks` / `load_all`
    on the sources inside methods when you need tensors.
    """
    align = (
        alignment
        if isinstance(alignment, Alignment)
        else get_alignment(alignment)
    )
    return DiffPair(
        a=ActivationSource(Path(path_a), capture, tags_a or {}),
        b=ActivationSource(Path(path_b), capture, tags_b or {}),
        alignment=align,
        relationship=relationship,
    )
