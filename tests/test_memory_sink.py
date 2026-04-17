from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import ActivationRecord


def _make_record(name: str, value: float, tags: dict[str, str]) -> ActivationRecord:
    return ActivationRecord(name=name, tensor=torch.tensor([value]), metadata_tags=tags)


def test_write_preserves_order() -> None:
    sink = MemoryActivationSink()
    with sink:
        sink.write(_make_record("encoder.0", 1.0, {"step": "0"}))
        sink.write(_make_record("encoder.1", 2.0, {"step": "1"}))
        sink.write(_make_record("encoder.2", 3.0, {"step": "2"}))

    assert [r.name for r in sink.records] == ["encoder.0", "encoder.1", "encoder.2"]
    assert [r.tensor.item() for r in sink.records] == [1.0, 2.0, 3.0]


def test_write_after_close_raises() -> None:
    sink = MemoryActivationSink()
    with sink:
        sink.write(_make_record("m", 0.0, {}))
    with pytest.raises(RuntimeError, match="closed"):
        sink.write(_make_record("m", 0.0, {}))


def test_per_cell_accepts_numpy_object_strings() -> None:
    # Per the v3 schema, per_cell may hold np.ndarray (incl. dtype=object
    # string arrays) in addition to torch tensors. ActivationRecord's
    # post_init uses .shape[0] which works for both.
    sink = MemoryActivationSink()
    pert = np.array(["ctrl", "ETS2", "CNN1+ETS2"], dtype=object)
    rec = ActivationRecord(
        name="enc",
        tensor=torch.zeros(3, 4),
        metadata_tags={"phase": "predict"},
        per_cell={
            "pert": pert,
            "cell_id": torch.arange(3),
        },
    )
    with sink:
        sink.write(rec)

    assert len(sink.records) == 1
    out = sink.records[0].per_cell
    assert list(out["pert"]) == ["ctrl", "ETS2", "CNN1+ETS2"]
    assert torch.equal(out["cell_id"], torch.arange(3))


def test_per_cell_ndarray_length_mismatch_rejected() -> None:
    # Length mismatch on a numpy-valued per_cell entry still raises, same
    # contract as with torch tensors.
    with pytest.raises(ValueError, match="per_cell"):
        ActivationRecord(
            name="enc",
            tensor=torch.zeros(3, 4),
            metadata_tags={},
            per_cell={"pert": np.array(["a", "b"], dtype=object)},
        )
