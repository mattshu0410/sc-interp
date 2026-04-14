from __future__ import annotations

import pytest
import torch

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import ActivationRecord


def _make_record(name: str, value: float, tags: dict[str, str]) -> ActivationRecord:
    return ActivationRecord(name=name, tensor=torch.tensor([value]), metadata_tags=tags)


def test_write_preserves_order_and_isolates_tags() -> None:
    sink = MemoryActivationSink()
    tags_a = {"phase": "predict", "step": "0"}
    tags_b = {"phase": "predict", "step": "1"}
    tags_c = {"phase": "predict", "step": "2"}

    with sink:
        sink.write(_make_record("encoder.0", 1.0, tags_a))
        sink.write(_make_record("encoder.1", 2.0, tags_b))
        sink.write(_make_record("encoder.2", 3.0, tags_c))
        tags_a["step"] = "MUTATED"
        tags_b["phase"] = "MUTATED"

    assert [r.name for r in sink.records] == ["encoder.0", "encoder.1", "encoder.2"]
    assert [r.tensor.item() for r in sink.records] == [1.0, 2.0, 3.0]
    assert sink.records[0].metadata_tags == {"phase": "predict", "step": "0"}
    assert sink.records[1].metadata_tags == {"phase": "predict", "step": "1"}
    assert sink.records[2].metadata_tags == {"phase": "predict", "step": "2"}


def test_write_after_close_raises() -> None:
    sink = MemoryActivationSink()
    with sink:
        sink.write(_make_record("m", 0.0, {}))
    with pytest.raises(RuntimeError, match="closed"):
        sink.write(_make_record("m", 0.0, {}))
