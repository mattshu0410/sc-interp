from __future__ import annotations

import torch
import torch.nn as nn
from nnsight import NNsight

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import HookManager


class _CallsTwice(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fuse = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        a = self.fuse(x)
        b = self.fuse(y)
        return a + b


def test_multi_invocation_captures_last_call() -> None:
    torch.manual_seed(0)
    net = _CallsTwice(4)
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    x = torch.randn(1, 4)
    y = torch.randn(1, 4)

    with HookManager(
        nn_model,
        capture=[("fuse", lambda m: m.fuse.output)],
        sink=sink,
    ) as hm:
        with hm.trace(x, y):
            pass

    with torch.no_grad():
        second_call = net.fuse(y)

    assert len(sink.records) == 1
    assert torch.allclose(sink.records[0].tensor, second_call)


def test_capture_parent_gets_aggregate() -> None:
    torch.manual_seed(0)
    net = _CallsTwice(4)
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    x = torch.randn(1, 4)
    y = torch.randn(1, 4)

    with HookManager(
        nn_model,
        capture=[("root", lambda m: m.output)],
        sink=sink,
    ) as hm:
        with hm.trace(x, y):
            pass

    with torch.no_grad():
        expected = net(x, y)

    assert torch.allclose(sink.records[0].tensor, expected)
