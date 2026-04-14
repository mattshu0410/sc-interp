from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from nnsight import NNsight

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import ActivationRecord, HookManager


class _CountingSink:
    def __init__(self) -> None:
        self.records: list[ActivationRecord] = []
        self.closed = False

    def write(self, record: ActivationRecord) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.closed = True


class _FailOnNthSink(_CountingSink):
    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self._fail_on = fail_on

    def write(self, record: ActivationRecord) -> None:
        super().write(record)
        if len(self.records) == self._fail_on:
            raise RuntimeError("sink boom")


def test_many_traces_all_arrive_in_order() -> None:
    torch.manual_seed(0)
    nn_model = NNsight(nn.Linear(4, 4))
    sink = _CountingSink()
    x = torch.randn(1, 4)

    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        for i in range(50):
            hm.set_tag("step", str(i))
            with hm.trace(x):
                pass

    assert [r.metadata_tags["step"] for r in sink.records] == [str(i) for i in range(50)]
    assert sink.closed


def test_sink_write_failure_reraised_on_exit() -> None:
    torch.manual_seed(0)
    nn_model = NNsight(nn.Linear(4, 4))
    sink = _FailOnNthSink(fail_on=5)
    x = torch.randn(1, 4)

    with pytest.raises(RuntimeError, match="sink boom"):
        with HookManager(
            nn_model, capture=[("lin", lambda m: m.output)], sink=sink
        ) as hm:
            for _ in range(20):
                with hm.trace(x):
                    pass

    assert sink.closed
    assert len(sink.records) == 5


def test_body_exception_still_closes_sink_and_joins_writer() -> None:
    torch.manual_seed(0)
    nn_model = NNsight(nn.Linear(4, 4))
    sink = MemoryActivationSink()
    x = torch.randn(1, 4)

    with pytest.raises(RuntimeError, match="outer"):
        with HookManager(
            nn_model, capture=[("lin", lambda m: m.output)], sink=sink
        ) as hm:
            with hm.trace(x):
                pass
            raise RuntimeError("outer")

    assert sink._closed
    assert len(sink.records) == 1
    assert not hm._writer.is_alive()  # type: ignore[union-attr]