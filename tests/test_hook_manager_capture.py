from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from nnsight import NNsight

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import HookManager


class _Block(nn.Module):
    def __init__(self, d: int, nh: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, d * 2), nn.GELU(), nn.Linear(d * 2, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class _TinyNet(nn.Module):
    def __init__(self, vocab: int = 10, d: int = 8, nh: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([_Block(d, nh)])
        self.head = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def test_capture_faithful_multi_target() -> None:
    torch.manual_seed(0)
    net = _TinyNet()
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    ids = torch.tensor([[0, 1, 2, 3]])

    with sink, HookManager(
        nn_model,
        capture=[
            ("embed", lambda m: m.embed.output),
            ("head", lambda m: m.head.output),
            ("blocks.0.mlp.0", lambda m: m.blocks[0].mlp[0].output),
            ("blocks.0.attn", lambda m: m.blocks[0].attn.output[0]),
        ],
        sink=sink,
    ) as hm:
        with hm.trace(ids):
            pass

    with torch.no_grad():
        emb = net.embed(ids)
        h = net.blocks[0].ln1(emb)
        attn_out, _ = net.blocks[0].attn(h, h, h, need_weights=False)
        x = emb + attn_out
        mlp0 = net.blocks[0].mlp[0](net.blocks[0].ln2(x))
        x = x + net.blocks[0].mlp(net.blocks[0].ln2(x))
        head = net.head(x)

    by_name = {r.name: r.tensor for r in sink.records}
    assert set(by_name) == {"embed", "head", "blocks.0.mlp.0", "blocks.0.attn"}
    assert torch.allclose(by_name["embed"], emb)
    assert torch.allclose(by_name["head"], head)
    assert torch.allclose(by_name["blocks.0.mlp.0"], mlp0)
    assert torch.allclose(by_name["blocks.0.attn"], attn_out)


def test_capture_linear_exact() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(4, 4)
    nn_model = NNsight(lin)
    sink = MemoryActivationSink()
    x = torch.randn(2, 4)

    with sink, HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output)],
        sink=sink,
    ) as hm:
        with hm.trace(x):
            pass

    expected = x @ lin.weight.T + lin.bias
    assert len(sink.records) == 1
    assert sink.records[0].name == "lin"
    assert torch.allclose(sink.records[0].tensor, expected)


def test_bad_accessor_raises() -> None:
    lin = nn.Linear(4, 4)
    nn_model = NNsight(lin)
    sink = MemoryActivationSink()

    def bad(_m: object) -> object:
        raise ValueError("boom")

    with sink, HookManager(nn_model, capture=[("bad", bad)], sink=sink) as hm:
        with pytest.raises(ValueError, match="boom"):
            with hm.trace(torch.randn(1, 4)):
                pass

    assert sink.records == []
