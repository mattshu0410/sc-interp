from __future__ import annotations

import torch
import torch.nn as nn
from nnsight import NNsight

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import HookManager


def _toy() -> tuple[NNsight, torch.Tensor]:
    torch.manual_seed(0)
    lin = nn.Linear(4, 4)
    return NNsight(lin), torch.randn(1, 4)


def test_tags_snapshotted_per_trace() -> None:
    nn_model, x = _toy()
    sink = MemoryActivationSink()

    with sink, HookManager(
        nn_model, capture=[("lin", lambda m: m.output)], sink=sink
    ) as hm:
        for i in range(5):
            hm.set_tag("step", str(i))
            with hm.trace(x):
                pass

    assert len(sink.records) == 5
    assert [r.metadata_tags["step"] for r in sink.records] == ["0", "1", "2", "3", "4"]

    hm.current_tags["step"] = "MUTATED"
    assert [r.metadata_tags["step"] for r in sink.records] == ["0", "1", "2", "3", "4"]


def test_gate_filters_records() -> None:
    nn_model, x = _toy()
    sink = MemoryActivationSink()

    with sink, HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output)],
        sink=sink,
        gate=lambda t: int(t["step"]) % 2 == 1,
    ) as hm:
        for i in range(5):
            hm.set_tag("step", str(i))
            with hm.trace(x):
                pass

    assert [r.metadata_tags["step"] for r in sink.records] == ["1", "3"]
