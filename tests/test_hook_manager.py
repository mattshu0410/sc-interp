from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from nnsight import NNsight

from scripts.interp.hook_sinks import MemoryActivationSink
from scripts.interp.hooks import ActivationRecord, HookManager


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


class _CallsTwice(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.fuse = nn.Linear(d, d)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fuse(x) + self.fuse(y)


class _FailOnNthSink:
    def __init__(self, fail_on: int) -> None:
        self.records: list[ActivationRecord] = []
        self.closed = False
        self._fail_on = fail_on

    def write(self, record: ActivationRecord) -> None:
        self.records.append(record)
        if len(self.records) == self._fail_on:
            raise RuntimeError("sink boom")

    def batch_end(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _FailOnNthSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@pytest.fixture
def simple_lin() -> tuple[NNsight, torch.Tensor]:
    torch.manual_seed(0)
    return NNsight(nn.Linear(4, 4)), torch.randn(1, 4)


# -- Capture (Step 2) ---------------------------------------------------


def test_capture_faithful_multi_target() -> None:
    torch.manual_seed(0)
    net = _TinyNet()
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    ids = torch.tensor([[0, 1, 2, 3]])

    # nnsight 0.5's InterleavingTracer requires .save() calls in forward
    # execution order. Order below matches _TinyNet.forward: embed → attn
    # → mlp.0 → head.
    with HookManager(
        nn_model,
        capture=[
            ("embed", lambda m: m.embed.output),
            ("blocks.0.attn", lambda m: m.blocks[0].attn.output[0]),
            ("blocks.0.mlp.0", lambda m: m.blocks[0].mlp[0].output),
            ("head", lambda m: m.head.output),
        ],
        sink=sink,
    ) as hm:
        hm.run(ids)

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
    # HookManager.run wraps the trace in torch.no_grad, so captured tensors
    # are bit-identical to the no_grad replay. Any numerical drift signals
    # a capture-path bug (grad mode leak, dtype change, view-not-copy).
    assert torch.equal(by_name["embed"], emb)
    assert torch.equal(by_name["blocks.0.attn"], attn_out)
    assert torch.equal(by_name["blocks.0.mlp.0"], mlp0)
    assert torch.equal(by_name["head"], head)
    for t in by_name.values():
        assert t.device.type == "cpu"
        assert t.requires_grad is False
        assert t.dtype == torch.float32


def test_capture_linear_exact(simple_lin: tuple[NNsight, torch.Tensor]) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        hm.run(x)

    expected = x @ nn_model._model.weight.T + nn_model._model.bias
    assert len(sink.records) == 1
    assert torch.equal(sink.records[0].tensor, expected)


def test_bad_accessor_raises(simple_lin: tuple[NNsight, torch.Tensor]) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    def bad(_m: object) -> object:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        with HookManager(nn_model, capture=[("bad", bad)], sink=sink) as hm:
            hm.run(x)

    assert sink.records == []


# -- Tags + gate (Step 3) ----------------------------------------------


def test_tags_snapshotted_per_trace(simple_lin: tuple[NNsight, torch.Tensor]) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        for i in range(5):
            hm.set_tag("step", str(i))
            hm.run(x)

    assert [r.metadata_tags["step"] for r in sink.records] == ["0", "1", "2", "3", "4"]
    hm.current_tags["step"] = "MUTATED"
    assert [r.metadata_tags["step"] for r in sink.records] == ["0", "1", "2", "3", "4"]


def test_gate_filters_records(simple_lin: tuple[NNsight, torch.Tensor]) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output)],
        sink=sink,
        gate=lambda t: int(t["step"]) % 2 == 1,
    ) as hm:
        for i in range(5):
            hm.set_tag("step", str(i))
            hm.run(x)

    assert [r.metadata_tags["step"] for r in sink.records] == ["1", "3"]


# -- Exception propagation (Step 4) ------------------------------------


def test_sink_write_failure_propagates(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = _FailOnNthSink(fail_on=5)

    with pytest.raises(RuntimeError, match="sink boom"):
        with HookManager(
            nn_model, capture=[("lin", lambda m: m.output)], sink=sink
        ) as hm:
            for _ in range(20):
                hm.run(x)

    assert sink.closed
    assert len(sink.records) == 5


def test_body_exception_still_closes_sink(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with pytest.raises(RuntimeError, match="outer"):
        with HookManager(
            nn_model, capture=[("lin", lambda m: m.output)], sink=sink
        ) as hm:
            hm.run(x)
            raise RuntimeError("outer")

    assert sink._closed
    assert len(sink.records) == 1


# -- Hardening: dtype, aliasing, close-masking, empty capture -----------


def test_capture_dtype_downcasts_to_fp16(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output)],
        sink=sink,
        capture_dtype=torch.float16,
    ) as hm:
        hm.run(x)

    rec = sink.records[0]
    assert rec.tensor.dtype == torch.float16
    expected = (x @ nn_model._model.weight.T + nn_model._model.bias).to(torch.float16)
    assert torch.equal(rec.tensor, expected)


def test_stored_tensor_does_not_alias_source(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        hm.run(x)
        # Mutate the captured tensor in place; re-running should produce a
        # fresh, correct tensor (no shared storage with the in-place-mutated
        # earlier record).
        sink.records[0].tensor.zero_()
        hm.run(x)

    expected = x @ nn_model._model.weight.T + nn_model._model.bias
    assert torch.all(sink.records[0].tensor == 0)
    assert torch.equal(sink.records[1].tensor, expected)


def test_exit_preserves_original_exception_when_close_fails(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin

    class _BadCloseSink:
        def __init__(self) -> None:
            self.records: list[ActivationRecord] = []

        def write(self, record: ActivationRecord) -> None:
            self.records.append(record)

        def batch_end(self) -> None:
            pass

        def close(self) -> None:
            raise RuntimeError("close boom")

        def __enter__(self) -> _BadCloseSink:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    sink = _BadCloseSink()
    # The original RuntimeError("outer") must surface — not "close boom".
    with pytest.raises(RuntimeError, match="outer"):
        with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
            hm.run(x)
            raise RuntimeError("outer")


def test_exit_surfaces_close_error_when_no_prior_exception(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin

    class _BadCloseSink:
        def write(self, record: ActivationRecord) -> None:
            pass

        def batch_end(self) -> None:
            pass

        def close(self) -> None:
            raise RuntimeError("close boom")

        def __enter__(self) -> _BadCloseSink:
            return self

        def __exit__(self, *exc: object) -> None:
            pass

    with pytest.raises(RuntimeError, match="close boom"):
        with HookManager(
            nn_model, capture=[("lin", lambda m: m.output)], sink=_BadCloseSink()
        ) as hm:
            hm.run(x)


# -- Multi-invocation (Step 5) -----------------------------------------


def test_multi_invocation_captures_first_call() -> None:
    # Under nnsight 0.5, `.output.save()` on a submodule called N times per
    # forward captures the FIRST invocation. (This differs from 0.3, which
    # captured the last — downstream docs and runners should use the
    # iterator API `.output.all().save()` when every invocation is needed.)
    torch.manual_seed(0)
    net = _CallsTwice(4)
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    x, y = torch.randn(1, 4), torch.randn(1, 4)

    with HookManager(nn_model, capture=[("fuse", lambda m: m.fuse.output)], sink=sink) as hm:
        hm.run(x, y)

    with torch.no_grad():
        first_call = net.fuse(x)
        second_call = net.fuse(y)
    captured = sink.records[0].tensor
    assert torch.equal(captured, first_call)
    # Guard against a last-call regression passing by coincidence: ensure the
    # capture is NOT the second invocation.
    assert not torch.allclose(captured, second_call)


def test_forward_order_violation_surfaces_error() -> None:
    # nnsight 0.5 raises when .save() calls are issued out of forward order.
    # The message varies by nnsight version; match substring "order" to
    # pin down the category instead of accepting any Exception.
    torch.manual_seed(0)
    net = _TinyNet()
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()

    with HookManager(
        nn_model,
        capture=[
            ("head", lambda m: m.head.output),
            ("embed", lambda m: m.embed.output),
        ],
        sink=sink,
    ) as hm:
        with pytest.raises(Exception, match="(?i)order"):
            hm.run(torch.tensor([[0, 1, 2, 3]]))


def test_kwargs_passthrough() -> None:
    # The kwarg `scale` is applied inside forward after a post-scale module;
    # capturing `post.output` proves the kwarg actually reached forward.
    # A regression that dropped kwargs would scale by the default 1.0 and
    # the captured `post` activation would mismatch the scale=2.0 reference.
    class _KwargsNet(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.pre = nn.Linear(d, d)
            self.post = nn.Linear(d, d)

        def forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
            return self.post(self.pre(x) * scale)

    torch.manual_seed(0)
    net = _KwargsNet(4)
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    x = torch.randn(2, 4)

    with HookManager(nn_model, capture=[("post", lambda m: m.post.output)], sink=sink) as hm:
        hm.run(x, scale=2.0)

    with torch.no_grad():
        expected = net.post(net.pre(x) * 2.0)
        default_scale = net.post(net.pre(x) * 1.0)
    captured = sink.records[0].tensor
    assert torch.equal(captured, expected)
    # Negative control: a regression dropping kwargs would produce `default_scale`.
    assert not torch.allclose(captured, default_scale)


def test_asymmetric_shapes_preserved() -> None:
    # Non-square dims (batch=3, seq=7, d=16, vocab=17) catch dim-swap
    # regressions that a (d, d) fixture would miss by symmetry.
    torch.manual_seed(0)
    net = _TinyNet(vocab=17, d=16, nh=2)
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    ids = torch.randint(0, 17, (3, 7))

    with HookManager(
        nn_model,
        capture=[
            ("embed", lambda m: m.embed.output),
            ("head", lambda m: m.head.output),
        ],
        sink=sink,
    ) as hm:
        hm.run(ids)

    with torch.no_grad():
        emb = net.embed(ids)
        h = net.blocks[0].ln1(emb)
        attn_out, _ = net.blocks[0].attn(h, h, h, need_weights=False)
        x = emb + attn_out
        x = x + net.blocks[0].mlp(net.blocks[0].ln2(x))
        head = net.head(x)

    by_name = {r.name: r.tensor for r in sink.records}
    assert torch.equal(by_name["embed"], emb)
    assert torch.equal(by_name["head"], head)


# -- Phase A.1: run returns model output --------------------------------


def test_run_returns_tensor_output(simple_lin: tuple[NNsight, torch.Tensor]) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        out = hm.run(x)

    expected = x @ nn_model._model.weight.T + nn_model._model.bias
    assert isinstance(out, torch.Tensor)
    assert torch.equal(out, expected)


def test_run_returns_dict_output() -> None:
    class _DictNet(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.lin = nn.Linear(d, d)

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            y = self.lin(x)
            return {"mlm_output": y, "other": y * 2}

    torch.manual_seed(0)
    net = _DictNet(4)
    nn_model = NNsight(net)
    x = torch.randn(2, 4)
    sink = MemoryActivationSink()

    with HookManager(nn_model, capture=[("lin", lambda m: m.lin.output)], sink=sink) as hm:
        out = hm.run(x)

    # scGPT-style dict round-trip: post-trace-exit, `out` is a real dict with
    # real tensors, not a proxy. This is the contract run_scgpt relies on for
    # extracting "mlm_output" while the sink captures inner activations.
    assert isinstance(out, dict)
    assert set(out) == {"mlm_output", "other"}
    with torch.no_grad():
        expected = net.lin(x)
    assert torch.equal(out["mlm_output"], expected)
    assert torch.equal(out["other"], expected * 2)


def test_run_empty_capture_still_returns_output(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(nn_model, capture=[], sink=sink) as hm:
        out = hm.run(x)

    expected = x @ nn_model._model.weight.T + nn_model._model.bias
    assert torch.equal(out, expected)
    assert sink.records == []


def test_out_of_order_save_raises() -> None:
    # User captures must be listed in forward-execution order. Reversing them
    # (head before embed) triggers nnsight's OutOfOrderError inside the trace.
    # Pin the category by matching the documented substring.
    torch.manual_seed(0)
    net = _TinyNet()
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()

    with HookManager(
        nn_model,
        capture=[
            ("head", lambda m: m.head.output),
            ("embed", lambda m: m.embed.output),
        ],
        sink=sink,
    ) as hm:
        with pytest.raises(Exception, match="(?i)out of order"):
            hm.run(torch.tensor([[0, 1, 2, 3]]))


# -- Phase A.2: set_per_cell + auto-clear -------------------------------


def test_set_per_cell_populates_and_clears(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    ids = np.array([f"c{i}" for i in range(x.shape[0])], dtype=object)
    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        hm.set_per_cell({"cell_id": ids})
        hm.run(x)
        # No set_per_cell on this run — auto-clear means record.per_cell empty.
        hm.run(x)

    assert len(sink.records) == 2
    assert set(sink.records[0].per_cell) == {"cell_id"}
    np.testing.assert_array_equal(sink.records[0].per_cell["cell_id"], ids)
    assert sink.records[1].per_cell == {}


def test_set_per_cell_snapshot_isolated_from_caller_mutation(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        ids = np.array([f"c{i}" for i in range(x.shape[0])], dtype=object)
        hm.set_per_cell({"cell_id": ids})
        hm.run(x)

    # Mutating the dict the caller passed in must not corrupt the written
    # record. set_per_cell takes a defensive shallow copy of the dict itself.
    assert set(sink.records[0].per_cell) == {"cell_id"}


def test_set_per_cell_accepts_mixed_tensor_and_ndarray(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    # A per_cell dict may mix torch tensors (numeric cell_index) and numpy
    # arrays (string cell_id, pert labels) — this is how runners make h5
    # files self-describing without needing a torch string dtype.
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    cell_id = np.array([f"c{i}" for i in range(x.shape[0])], dtype=object)
    cell_index = torch.arange(x.shape[0])
    pert = np.array(["ctrl"] * x.shape[0], dtype=object)
    with HookManager(nn_model, capture=[("lin", lambda m: m.output)], sink=sink) as hm:
        hm.set_per_cell({
            "cell_id": cell_id,
            "cell_index": cell_index,
            "pert": pert,
        })
        hm.run(x)

    pc = sink.records[0].per_cell
    assert set(pc) == {"cell_id", "cell_index", "pert"}
    np.testing.assert_array_equal(pc["cell_id"], cell_id)
    assert torch.equal(pc["cell_index"], cell_index)
    assert list(pc["pert"]) == ["ctrl"] * x.shape[0]


# -- Phase A.2.5: cell_id contract --------------------------------------


def test_set_per_cell_rejects_missing_cell_id() -> None:
    hm = HookManager(nn_model=None, capture=[], sink=None)
    with pytest.raises(ValueError, match="set_per_cell requires a 'cell_id'"):
        hm.set_per_cell({"pert": np.array(["ctrl"], dtype=object)})


def test_set_per_cell_rejects_torch_int_cell_id() -> None:
    hm = HookManager(nn_model=None, capture=[], sink=None)
    with pytest.raises(TypeError, match="cell_id must be a string-dtype"):
        hm.set_per_cell({"cell_id": torch.arange(3)})


def test_set_per_cell_rejects_numpy_int_cell_id() -> None:
    hm = HookManager(nn_model=None, capture=[], sink=None)
    with pytest.raises(TypeError, match="cell_id must be a string-dtype"):
        hm.set_per_cell({"cell_id": np.arange(3)})


def test_set_per_cell_rejects_within_batch_duplicates() -> None:
    hm = HookManager(nn_model=None, capture=[], sink=None)
    with pytest.raises(ValueError, match="duplicates within batch"):
        hm.set_per_cell({"cell_id": np.array(["a", "a", "b"], dtype=object)})


def test_set_per_cell_rejects_across_batch_duplicates() -> None:
    hm = HookManager(nn_model=None, capture=[], sink=None)
    hm.set_per_cell({"cell_id": np.array(["a", "b"], dtype=object)})
    with pytest.raises(ValueError, match="duplicates entries from earlier batches"):
        hm.set_per_cell({"cell_id": np.array(["b", "c"], dtype=object)})


# -- Phase A.3: no_grad ctor param --------------------------------------


def test_layout_propagates_from_capture_spec(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    # 3-tuple CaptureSpec form: layout travels with the target that owns it.
    with HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output, "BD")],
        sink=sink,
    ) as hm:
        hm.run(x)

    assert sink.records[0].layout == "BD"


def test_gate_false_still_returns_output(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    # Pins the documented contract: gate filters sink writes, but the forward
    # result is always returned so predict loops stay uninterrupted.
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output)],
        sink=sink,
        gate=lambda _tags: False,
    ) as hm:
        out = hm.run(x)

    expected = x @ nn_model._model.weight.T + nn_model._model.bias
    assert sink.records == []
    assert torch.equal(out, expected)


def test_no_grad_false_preserves_autograd(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    nn_model, x = simple_lin
    sink = MemoryActivationSink()

    with HookManager(
        nn_model, capture=[("lin", lambda m: m.output)], sink=sink, no_grad=False
    ) as hm:
        out = hm.run(x)

    # Weights require grad → forward produces a grad-carrying tensor when no_grad=False.
    # Returned output retains requires_grad. Captured record is detached by design
    # (sink-stored tensors must be standalone for disk round-trip), so we only
    # assert on the returned output.
    assert out.requires_grad is True


def test_capture_parent_gets_aggregate() -> None:
    torch.manual_seed(0)
    net = _CallsTwice(4)
    net.eval()
    nn_model = NNsight(net)
    sink = MemoryActivationSink()
    x, y = torch.randn(1, 4), torch.randn(1, 4)

    with HookManager(nn_model, capture=[("root", lambda m: m.output)], sink=sink) as hm:
        hm.run(x, y)

    with torch.no_grad():
        expected = net(x, y)
    assert torch.equal(sink.records[0].tensor, expected)


def test_hook_manager_calls_batch_end_once_per_run(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    # Sharding cadence depends on a one-call-per-run contract regardless of
    # how many per-target writes fired inside the run.
    nn_model, x = simple_lin
    sink = MemoryActivationSink()
    with HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output), ("lin2", lambda m: m.output)],
        sink=sink,
    ) as hm:
        for _ in range(3):
            hm.run(x)
    assert sink.batch_end_count == 3
    # Sanity: 3 runs × 2 targets = 6 writes, independent of batch_end count.
    assert len(sink.records) == 6


def test_hook_manager_calls_batch_end_when_gate_drops(
    simple_lin: tuple[NNsight, torch.Tensor],
) -> None:
    # Gate-dropped batches are still batch boundaries — skipping batch_end
    # on them would make shard rotation drift relative to the caller loop.
    nn_model, x = simple_lin
    sink = MemoryActivationSink()
    with HookManager(
        nn_model,
        capture=[("lin", lambda m: m.output)],
        sink=sink,
        gate=lambda _tags: False,
    ) as hm:
        for _ in range(4):
            hm.run(x)
    assert sink.batch_end_count == 4
    assert len(sink.records) == 0