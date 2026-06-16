"""Build a DiffPair from h5 activation paths."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py

from scripts.diffing.alignment import Alignment, get_alignment
from scripts.diffing.base import ActivationSource, DiffPair, Relationship, _shard_files
from scripts.interp.hook_sinks import group_path


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

    When both sides expose a `cell_id` label column, the longer side is
    truncated to the shorter (`max_rows`) and the shared prefix is
    verified element-wise. Mismatch raises with the first divergent row;
    no silent partial diff. Sides without `cell_id` (legacy shards) fall
    back to the row-count equality check in `DiffPair.__post_init__`.

    `iter_chunks` / `load_all` on the sources inside methods are what
    actually read tensors.
    """
    align = (
        alignment
        if isinstance(alignment, Alignment)
        else get_alignment(alignment)
    )
    a = ActivationSource(Path(path_a), capture, tags_a or {})
    b = ActivationSource(Path(path_b), capture, tags_b or {})

    a_ids = _read_cell_ids(a)
    b_ids = _read_cell_ids(b)
    if a_ids is not None and b_ids is not None:
        k = min(len(a_ids), len(b_ids))
        for i in range(k):
            if a_ids[i] != b_ids[i]:
                raise ValueError(
                    f"cell_id prefix mismatch at row {i}: "
                    f"a={a_ids[i]!r} b={b_ids[i]!r}. Both sides must be "
                    "prefix-aligned (same loader, same seed). For "
                    "non-prefix overlap support see issue tracking "
                    "general align_by."
                )
        a = replace(a, max_rows=k)
        b = replace(b, max_rows=k)
        print(
            f"==> aligned by cell_id prefix: a={len(a_ids)}, b={len(b_ids)}, "
            f"truncated to k={k}"
        )

    return DiffPair(a=a, b=b, alignment=align, relationship=relationship)


def _read_cell_ids(src: ActivationSource) -> list | None:
    """Walk shards reading just the cell_id label column. Return None if
    no shard has it (legacy data without the cell_id contract)."""
    label_path = f"{group_path(src.capture, src.tags)}/labels/cell_id"
    ids: list = []
    found = False
    for sf in _shard_files(src.path):
        with h5py.File(sf, "r") as f:
            if label_path not in f:
                continue
            found = True
            arr = f[label_path][:]
            if arr.dtype == object:
                ids.extend(
                    x.decode() if isinstance(x, bytes) else x for x in arr
                )
            else:
                ids.extend(arr.tolist())
    return ids if found else None
