from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
flax = pytest.importorskip("flax")
import flax.linen as nn  # noqa: E402

from scripts.interp.cellflow_extractor import CellflowActivationCapture  # noqa: E402
from scripts.interp.hook_sinks import MemoryActivationSink  # noqa: E402


class ToyVF(nn.Module):
    hidden: int = 8

    def setup(self):
        self.time_linear = nn.Dense(self.hidden)
        self.x_linear = nn.Dense(self.hidden)
        self.cond_linear = nn.Dense(self.hidden)
        self.decoder_linear = nn.Dense(self.hidden)
        self.output_linear = nn.Dense(self.hidden)

    def __call__(self, t, x_t, cond, encoder_noise, train: bool = False):
        t_enc = self.time_linear(jnp.asarray([t, t * 2.0, t * 3.0])[None, :])
        x_enc = self.x_linear(x_t)
        # Mirror cellflow's "cond comes out (1,D), tile to (B,D)" shape.
        cond_pre = self.cond_linear(cond["c"])
        if cond_pre.shape[0] != x_t.shape[0]:
            cond_mean = jnp.tile(cond_pre, (x_t.shape[0], 1))
        else:
            cond_mean = cond_pre
        if t_enc.shape[0] != x_t.shape[0]:
            t_enc = jnp.tile(t_enc, (x_t.shape[0], 1))

        self.sow("intermediates", "time_enc", t_enc)
        self.sow("intermediates", "x_enc", x_enc)
        self.sow("intermediates", "condition_mean", cond_mean)

        pre_decoder = jnp.concatenate([t_enc, x_enc, cond_mean], axis=-1)
        self.sow("intermediates", "pre_decoder", pre_decoder)

        dec = self.decoder_linear(pre_decoder)
        self.sow("intermediates", "decoder", dec)

        out = self.output_linear(dec)
        self.sow("intermediates", "output", out)
        return out, cond_mean, jnp.zeros_like(cond_mean)


@pytest.fixture
def vf_and_params():
    model = ToyVF(hidden=8)
    rng = jax.random.PRNGKey(0)
    x = jnp.zeros((4, 6))
    cond = {"c": jnp.zeros((1, 5))}
    encoder_noise = jnp.zeros((1, 8))
    variables = model.init(rng, jnp.asarray(0.0, dtype=jnp.float32), x, cond, encoder_noise)
    return model, variables["params"]


CAPTURE_NAMES = ["time_enc", "x_enc", "condition_mean", "pre_decoder", "decoder", "output"]


def test_capture_faithful_fixed_t(vf_and_params):
    model, params = vf_and_params
    sink = MemoryActivationSink()
    timesteps = (0.0, 0.25, 0.5)
    B = 4

    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=CAPTURE_NAMES,
        sink=sink,
        timesteps=timesteps,
    )
    cap.set_tag("phase", "predict")

    rng = jax.random.PRNGKey(1)
    x1 = jax.random.normal(rng, (B, 6))
    x2 = jax.random.normal(jax.random.PRNGKey(2), (B, 6))
    cond = {"c": jnp.ones((1, 5))}
    enc_noise = jnp.zeros((1, 8))

    # Manually replay each (batch, t) to get ground-truth sow values.
    for batch_idx, batch_x in enumerate((x1, x2)):
        ids = np.array(
            [f"b{batch_idx}c{i}" for i in range(batch_x.shape[0])],
            dtype=object,
        )
        cap.set_per_cell({"cell_id": ids})
        cap.run(batch_x, cond, enc_noise)

    # Total record count = n_batches * n_timesteps * n_names
    assert len(sink.records) == 2 * len(timesteps) * len(CAPTURE_NAMES)

    # Element-wise faithful capture vs manual replay of the forward.
    expected_by_key: dict[tuple[str, str, int], torch.Tensor] = {}
    batch_idx = 0
    for batch_x in (x1, x2):
        for t in timesteps:
            _, variables = model.apply(
                {"params": params},
                jnp.asarray(t, dtype=jnp.float32),
                batch_x,
                cond,
                enc_noise,
                train=False,
                mutable=["intermediates"],
            )
            inters = jax.device_get(variables["intermediates"])
            for name in CAPTURE_NAMES:
                arr = np.asarray(inters[name][0])
                expected_by_key[(name, f"{t:.2f}", batch_idx)] = torch.from_numpy(arr)
        batch_idx += 1

    # Assert each record matches expected under the same (name, t, batch).
    # Records are written in (batch, t, name) order.
    i = 0
    for b in range(2):
        for t in timesteps:
            for name in CAPTURE_NAMES:
                rec = sink.records[i]
                assert rec.name == name
                assert rec.metadata_tags["t"] == f"{t:.2f}"
                assert rec.metadata_tags["phase"] == "predict"
                exp = expected_by_key[(name, f"{t:.2f}", b)]
                assert torch.allclose(rec.tensor, exp.to(rec.tensor.dtype), atol=0.0)
                i += 1


def test_per_cell_cleared_between_runs(vf_and_params):
    model, params = vf_and_params
    sink = MemoryActivationSink()
    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=["output"],
        sink=sink,
        timesteps=(0.0, 0.5),
    )
    x = jnp.zeros((3, 6))
    cond = {"c": jnp.zeros((1, 5))}
    enc_noise = jnp.zeros((1, 8))

    ids = np.array(["c0", "c1", "c2"], dtype=object)
    cap.set_per_cell({"cell_id": ids})
    cap.run(x, cond, enc_noise)

    # All records from run #1 share the same cell_id (per-run snapshot, not
    # per-timestep re-read). Catches a bug where set_per_cell fired mid-run.
    first_run = sink.records[:2]
    assert len(first_run) == 2
    for rec in first_run:
        np.testing.assert_array_equal(rec.per_cell["cell_id"], ids)

    # Run again without setting per_cell — should be empty, not stale.
    cap.run(x, cond, enc_noise)
    second_run = sink.records[2:]
    assert len(second_run) == 2
    for rec in second_run:
        assert rec.per_cell == {}


def test_gate_filters_timesteps(vf_and_params):
    model, params = vf_and_params
    sink = MemoryActivationSink()
    # Keep only odd quarter-steps: t=0.25 and t=0.75.
    gate = lambda tags: int(round(float(tags["t"]) * 4)) % 2 == 1
    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=["output"],
        sink=sink,
        timesteps=(0.0, 0.25, 0.5, 0.75, 1.0),
        gate=gate,
    )
    x = jnp.zeros((2, 6))
    cond = {"c": jnp.zeros((1, 5))}
    enc_noise = jnp.zeros((1, 8))
    cap.run(x, cond, enc_noise)
    # 2 surviving timesteps × 1 capture name
    assert len(sink.records) == 2
    kept = sorted(r.metadata_tags["t"] for r in sink.records)
    assert kept == ["0.25", "0.75"]


def test_gate_non_bool_raises(vf_and_params):
    model, params = vf_and_params
    sink = MemoryActivationSink()
    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=["output"],
        sink=sink,
        timesteps=(0.0,),
        gate=lambda tags: None,  # type: ignore[return-value]
    )
    with pytest.raises(TypeError, match="gate returned"):
        cap.run(jnp.zeros((2, 6)), {"c": jnp.zeros((1, 5))}, jnp.zeros((1, 8)))


def test_capture_jax_materialization_no_traced_leakage(vf_and_params):
    model, params = vf_and_params
    sink = MemoryActivationSink()
    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=["output"],
        sink=sink,
        timesteps=(0.25,),
    )
    cap.run(jnp.ones((2, 6)), {"c": jnp.zeros((1, 5))}, jnp.zeros((1, 8)))
    rec = sink.records[0]
    assert isinstance(rec.tensor, torch.Tensor)
    # A jax Tracer would refuse .numpy() — this is the concrete-tensor check.
    _ = rec.tensor.numpy()


def test_t_passed_to_apply_is_rank_2_matching_batch(vf_and_params):
    model, params = vf_and_params
    seen_t_shapes: list[tuple[int, ...]] = []

    class ShapeSpyVF(type(model)):  # shim wrapping the toy's apply
        pass

    # Wrap vf.apply to capture the `t` arg's shape for every call.
    orig_apply = model.apply

    def spy_apply(*args, **kwargs):
        # Flax `apply(variables, t, x, cond, encoder_noise, ...)` positional
        # layout — record t's shape pre-forward.
        t_arg = args[1]
        seen_t_shapes.append(tuple(getattr(t_arg, "shape", ())))
        return orig_apply(*args, **kwargs)

    model.apply = spy_apply  # type: ignore[assignment]
    try:
        cap = CellflowActivationCapture(
            vf_module=model,
            params=params,
            capture_names=["output"],
            sink=MemoryActivationSink(),
            timesteps=(0.0, 0.25, 0.5),
        )
        B = 7
        cap.run(jnp.zeros((B, 6)), {"c": jnp.zeros((1, 5))}, jnp.zeros((1, 8)))
    finally:
        model.apply = orig_apply  # type: ignore[assignment]

    # Every apply call got a rank-2 `(B, 1)` t. Scalar-t regression would
    # break cellflow's real concat at _velocity_field.py:219.
    assert seen_t_shapes == [(B, 1)] * 3


def test_capture_dtype_fp16_actually_casts(vf_and_params):
    model, params = vf_and_params
    sink_fp16 = MemoryActivationSink()
    sink_fp32 = MemoryActivationSink()
    x = jnp.ones((2, 6))
    cond = {"c": jnp.zeros((1, 5))}
    enc = jnp.zeros((1, 8))

    cap16 = CellflowActivationCapture(
        vf_module=model, params=params, capture_names=CAPTURE_NAMES,
        sink=sink_fp16, timesteps=(0.0,), capture_dtype=torch.float16,
    )
    cap32 = CellflowActivationCapture(
        vf_module=model, params=params, capture_names=CAPTURE_NAMES,
        sink=sink_fp32, timesteps=(0.0,), capture_dtype=torch.float32,
    )
    cap16.run(x, cond, enc)
    cap32.run(x, cond, enc)

    # dtype AND value: the cast must happen AND produce the same bit pattern
    # as a downstream fp32→fp16 round-trip of the same activation. Without
    # the value check, a buggy `.to(dtype)` that returns fp32 with a
    # mislabelled header would slip through.
    assert len(sink_fp16.records) == len(sink_fp32.records) == len(CAPTURE_NAMES)
    for r16, r32 in zip(sink_fp16.records, sink_fp32.records):
        assert r16.tensor.dtype == torch.float16
        assert r32.tensor.dtype == torch.float32
        assert torch.equal(r16.tensor, r32.tensor.to(torch.float16))


def test_records_written_in_timestep_order_per_batch(vf_and_params):
    model, params = vf_and_params
    sink = MemoryActivationSink()
    timesteps = (0.0, 0.25, 0.5)
    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=["time_enc", "output"],
        sink=sink,
        timesteps=timesteps,
    )
    cap.run(jnp.zeros((2, 6)), {"c": jnp.zeros((1, 5))}, jnp.zeros((1, 8)))
    # Expected ordering: for each t in timesteps, emit records in
    # capture_names order.
    expected = []
    for t in timesteps:
        for name in ["time_enc", "output"]:
            expected.append((name, f"{t:.2f}"))
    actual = [(r.name, r.metadata_tags["t"]) for r in sink.records]
    assert actual == expected


def test_cellflow_extractor_calls_batch_end_once_per_run(vf_and_params):
    # Sharding cadence depends on one batch_end per run(), regardless of
    # how many records (timesteps × names) the run emits. Gate-dropped
    # runs must also fire batch_end so the shard counter stays aligned
    # with the caller's batch loop.
    model, params = vf_and_params
    sink = MemoryActivationSink()
    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=["time_enc", "output"],
        sink=sink,
        timesteps=(0.0, 0.5, 1.0),
        gate=lambda tags: tags["t"] != "0.50",  # drop t=0.5 only
    )
    for _ in range(3):
        cap.run(jnp.zeros((2, 6)), {"c": jnp.zeros((1, 5))}, jnp.zeros((1, 8)))
    assert sink.batch_end_count == 3
    # 3 runs × (3 timesteps - 1 gated out) × 2 names = 12 records.
    assert len(sink.records) == 12


def test_cellflow_extractor_calls_batch_end_when_all_gated(vf_and_params):
    model, params = vf_and_params
    sink = MemoryActivationSink()
    cap = CellflowActivationCapture(
        vf_module=model,
        params=params,
        capture_names=["output"],
        sink=sink,
        timesteps=(0.0, 0.5),
        gate=lambda _tags: False,
    )
    for _ in range(2):
        cap.run(jnp.zeros((2, 6)), {"c": jnp.zeros((1, 5))}, jnp.zeros((1, 8)))
    assert sink.batch_end_count == 2
    assert len(sink.records) == 0
