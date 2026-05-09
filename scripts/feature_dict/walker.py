"""Stream cells through a FeatureSource, preserving per-token gene IDs.

Unlike `scripts.diffing.methods.crosscoder.dataloader.iter_pair_samples`,
this walker does NOT broadcast cell-level labels across token positions:
``gene_dataset_ids`` is kept as ``(n_cells, T)`` so each token can be
mapped back to its gene symbol for enrichment analysis.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

from scripts.diffing.base import DiffPair
from scripts.feature_dict.sources import FeatureSource


@dataclass(frozen=True)
class Chunk:
    """One streaming batch of crosscoder codes with per-cell + per-token labels.

    Layout: codes is ``(n, T, dict_size)``; cell-level labels are arrays of
    length ``n``; ``gene_dataset_ids`` is ``(n, T)`` indexing into the run's
    ``meta/gene_symbols`` (resolve via :func:`resolve_gene_symbols`).
    """
    codes: torch.Tensor                  # (n, T, dict_size), float32, on `device`
    cell_id: np.ndarray                  # (n,) dtype=object
    pert: np.ndarray                     # (n,) dtype=object
    cell_line: np.ndarray                # (n,) dtype=object — parsed from cell_id
    gene_dataset_ids: np.ndarray         # (n, T) int64 indices into gene_symbols


def _parse_cell_line(cell_id: np.ndarray) -> np.ndarray:
    """Parse the cell-line suffix from ``<barcode>-<batch>-<cell_line>``."""
    return np.array([c.split("-")[-1] for c in cell_id], dtype=object)


def walk_pair(
    pair: DiffPair,
    source: FeatureSource,
    *,
    chunk_rows: int = 32,
    device: str = "cuda",
) -> Iterator[Chunk]:
    """Yield aligned ``Chunk``s of crosscoder codes for cells in `pair`.

    Streams `pair.a` (base) and `pair.b` (ESM) shard-by-shard, applies the
    pair's alignment to enforce cell-id ordering, encodes the
    ``(n*T, D)`` flat tensors via `source`, then reshapes back to
    ``(n, T, dict_size)``.

    `chunk_rows` is in cells, not samples. Memory cost per yield ≈
    ``chunk_rows * T * dict_size * 4 bytes`` on `device`. With T=1536 and
    dict_size=4096, 32 cells ≈ 800 MB on GPU.
    """
    sides = source.needed_sides
    if len(sides) != 2:
        # Walker is 2-sided. SAE-style 1-sided sources need a sibling walker.
        raise ValueError(
            f"walk_pair expects a 2-sided source; got needed_sides={sides}. "
            "Use a 1-sided walker for SAE sources."
        )
    side_a_key, side_b_key = sides

    a_iter = pair.a.iter_chunks(chunk_rows)
    b_iter = pair.b.iter_chunks(chunk_rows)
    for (a_chunk, a_labels), (b_chunk, _b_labels) in zip(a_iter, b_iter):
        a_aligned, b_aligned = pair.alignment.apply(a_chunk, b_chunk)
        if a_aligned.ndim != 3:
            raise ValueError(
                f"walk_pair needs BTD activations; got ndim={a_aligned.ndim}"
            )
        n, t, d = a_aligned.shape
        codes_flat = source.encode({
            side_a_key: a_aligned.reshape(n * t, d),
            side_b_key: b_aligned.reshape(n * t, d),
        })
        codes = codes_flat.reshape(n, t, source.dict_size)
        cell_id = a_labels["cell_id"]
        yield Chunk(
            codes=codes,
            cell_id=cell_id,
            pert=a_labels["pert"],
            cell_line=_parse_cell_line(cell_id),
            gene_dataset_ids=a_labels["gene_dataset_ids"],
        )


def resolve_gene_symbols(activation_path) -> np.ndarray:
    """Read ``meta/gene_symbols`` from a capture folder's first shard.

    Returns ``(V,)`` dtype=object string array; ``gene_dataset_ids[c, t]``
    indexes into this.
    """
    import h5py
    from pathlib import Path
    p = Path(activation_path)
    shards = sorted(p.glob("shard-*.h5"))
    if not shards:
        raise FileNotFoundError(f"no shard-*.h5 in {p}")
    with h5py.File(shards[0]) as f:
        arr = f["meta/gene_symbols"][:]
    return np.array(
        [x.decode() if isinstance(x, bytes) else str(x) for x in arr], dtype=object
    )
