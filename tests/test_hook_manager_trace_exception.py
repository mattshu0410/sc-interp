from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from nnsight import NNsight

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import HookManager


def test_body_exception_propagates_and_writes_nothing() -> None:
    torch.manual_seed(0)
    nn_model = NNsight(nn.Linear(4, 4))
    sink = MemoryActivationSink()
    x = torch.randn(1, 4)

    with sink, HookManager(
        nn_model, capture=[("lin", lambda m: m.output)], sink=sink
    ) as hm:
        with pytest.raises(RuntimeError, match="user-raised"):
            with hm.trace(x):
                raise RuntimeError("user-raised")

    assert sink.records == []


def test_successful_traces_around_failing_one_are_recorded() -> None:
    torch.manual_seed(0)
    nn_model = NNsight(nn.Linear(4, 4))
    sink = MemoryActivationSink()
    x = torch.randn(1, 4)

    with sink, HookManager(
        nn_model, capture=[("lin", lambda m: m.output)], sink=sink
    ) as hm:
        hm.set_tag("step", "before")
        with hm.trace(x):
            pass

        hm.set_tag("step", "failing")
        with pytest.raises(RuntimeError):
            with hm.trace(x):
                raise RuntimeError("boom")

        hm.set_tag("step", "after")
        with hm.trace(x):
            pass

    assert [r.metadata_tags["step"] for r in sink.records] == ["before", "after"]
