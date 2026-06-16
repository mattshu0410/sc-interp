from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.diffing.pairs import load_pair
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


CAPTURE = "layer_0"


def _write(path: Path, tensor: torch.Tensor, cell_ids: list[str] | None) -> None:
    per_cell: dict = {}
    if cell_ids is not None:
        per_cell["cell_id"] = np.array(cell_ids, dtype=object)
    with H5ActivationSink(
        path, runner="t", dataset="t", split="t", capture_names=[CAPTURE]
    ) as sink:
        sink.write(
            ActivationRecord(
                name=CAPTURE,
                tensor=tensor,
                metadata_tags={},
                per_cell=per_cell,
                layout="BD",
            )
        )


def test_prefix_aligned_equal_length_passes(tmp_path: Path) -> None:
    ids = [f"c{i}" for i in range(10)]
    _write(tmp_path / "a.h5", torch.zeros(10, 4), ids)
    _write(tmp_path / "b.h5", torch.zeros(10, 4), ids)
    pair = load_pair(tmp_path / "a.h5", tmp_path / "b.h5", capture=CAPTURE)
    assert pair.a.row_count() == 10
    assert pair.b.row_count() == 10


def test_prefix_aligned_b_shorter_truncates_a(tmp_path: Path) -> None:
    ids_a = [f"c{i}" for i in range(10)]
    ids_b = [f"c{i}" for i in range(5)]
    _write(tmp_path / "a.h5", torch.zeros(10, 4), ids_a)
    _write(tmp_path / "b.h5", torch.zeros(5, 4), ids_b)
    pair = load_pair(tmp_path / "a.h5", tmp_path / "b.h5", capture=CAPTURE)
    assert pair.a.row_count() == 5
    assert pair.b.row_count() == 5
    a_chunks = list(pair.a.iter_chunks(chunk_rows=100))
    b_chunks = list(pair.b.iter_chunks(chunk_rows=100))
    assert sum(c.shape[0] for c, _ in a_chunks) == 5
    assert sum(c.shape[0] for c, _ in b_chunks) == 5


def test_non_prefix_overlap_raises_with_row(tmp_path: Path) -> None:
    _write(tmp_path / "a.h5", torch.zeros(5, 4), ["c0", "c1", "c2", "c3", "c4"])
    _write(tmp_path / "b.h5", torch.zeros(5, 4), ["c0", "c1", "c2", "X9", "c4"])
    with pytest.raises(ValueError, match=r"cell_id prefix mismatch at row 3"):
        load_pair(tmp_path / "a.h5", tmp_path / "b.h5", capture=CAPTURE)


def test_no_cell_id_falls_back_to_row_equality(tmp_path: Path) -> None:
    _write(tmp_path / "a.h5", torch.zeros(10, 4), cell_ids=None)
    _write(tmp_path / "b.h5", torch.zeros(12, 4), cell_ids=None)
    with pytest.raises(ValueError, match="row mismatch"):
        load_pair(tmp_path / "a.h5", tmp_path / "b.h5", capture=CAPTURE)


def test_one_side_has_cell_id_falls_back(tmp_path: Path) -> None:
    _write(tmp_path / "a.h5", torch.zeros(10, 4), [f"c{i}" for i in range(10)])
    _write(tmp_path / "b.h5", torch.zeros(10, 4), cell_ids=None)
    pair = load_pair(tmp_path / "a.h5", tmp_path / "b.h5", capture=CAPTURE)
    assert pair.a.row_count() == 10
    assert pair.b.row_count() == 10


def test_chunks_yielded_respect_max_rows(tmp_path: Path) -> None:
    ids_a = [f"c{i}" for i in range(7)]
    ids_b = [f"c{i}" for i in range(4)]
    _write(tmp_path / "a.h5", torch.arange(28, dtype=torch.float32).reshape(7, 4), ids_a)
    _write(tmp_path / "b.h5", torch.arange(16, dtype=torch.float32).reshape(4, 4), ids_b)
    pair = load_pair(tmp_path / "a.h5", tmp_path / "b.h5", capture=CAPTURE)
    a_acts = torch.cat([c for c, _ in pair.a.iter_chunks(chunk_rows=2)], dim=0)
    b_acts = torch.cat([c for c, _ in pair.b.iter_chunks(chunk_rows=2)], dim=0)
    assert a_acts.shape == (4, 4)
    assert b_acts.shape == (4, 4)
    torch.testing.assert_close(a_acts, torch.arange(16, dtype=torch.float32).reshape(4, 4))
