from __future__ import annotations

import importlib
import inspect

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
cellflow = pytest.importorskip("cellflow")

from cellflow.networks._velocity_field import (  # noqa: E402
    ConditionalVelocityField,
    GENOTConditionalVelocityField,
)

from scripts.interp import cellflow_probes  # noqa: E402
from scripts.interp.cellflow_probes import (  # noqa: E402
    patch_velocity_field,
    unpatch_velocity_field,
)


@pytest.fixture(autouse=True)
def _restore_patch_state():
    # Every test starts clean and leaves state clean: the monkey-patch is
    # process-global and would otherwise leak into neighbouring tests.
    unpatch_velocity_field()
    yield
    unpatch_velocity_field()


def _build_otfm_vf():
    return ConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(16, 16),
        decoder_dims=(16, 16),
        time_encoder_dims=(16, 16),
    )


def _build_genot_vf():
    return GENOTConditionalVelocityField(
        output_dim=5,
        max_combination_length=2,
        condition_embedding_dim=12,
        hidden_dims=(16, 16),
        decoder_dims=(16, 16),
        time_encoder_dims=(16, 16),
    )


def _init_inputs_otfm():
    rng = jax.random.PRNGKey(0)
    B = 4
    x = jnp.ones((B, 5))
    cond = {"pert1": jnp.ones((1, 2, 3))}
    encoder_noise = jnp.zeros((1, 12))
    # Rank-2 `(B, 1)` — scalar t would break concat at
    # _velocity_field.py:219 against rank-2 x_encoded.
    t = jnp.full((B, 1), 0.5, dtype=jnp.float32)
    return rng, t, x, cond, encoder_noise


def _init_inputs_genot():
    rng = jax.random.PRNGKey(0)
    B = 4
    x = jnp.ones((B, 5))
    x_0 = jnp.ones((B, 5)) * 0.5
    cond = {"pert1": jnp.ones((1, 2, 3))}
    encoder_noise = jnp.zeros((1, 12))
    t = jnp.full((B, 1), 0.5, dtype=jnp.float32)
    return rng, t, x, x_0, cond, encoder_noise


def test_patch_is_idempotent():
    patch_velocity_field()
    first_call = ConditionalVelocityField.__call__
    first_probed = getattr(ConditionalVelocityField, "__cellflow_probed__", None)
    first_original = getattr(ConditionalVelocityField, "_cellflow_original_call", None)

    patch_velocity_field()
    second_call = ConditionalVelocityField.__call__
    second_probed = getattr(ConditionalVelocityField, "__cellflow_probed__", None)
    second_original = getattr(ConditionalVelocityField, "_cellflow_original_call", None)

    assert first_probed is True
    assert second_probed is True
    # Second call must be a true no-op, not a re-wrap around the first patch.
    assert first_call is second_call
    # The saved original must be the pre-patch __call__ and must NOT be
    # overwritten by the second patch (else unpatch would restore the
    # patched version, wedging subsequent tests).
    assert first_original is not None
    assert first_original is second_original
    assert first_original is not first_call


def test_patch_after_reload_re_patches():
    patch_velocity_field()
    mod = importlib.reload(
        importlib.import_module("cellflow.networks._velocity_field")
    )
    fresh_cls = mod.ConditionalVelocityField
    # Fresh class has no attr — this is what "detect reload" means.
    assert not getattr(fresh_cls, "__cellflow_probed__", False)
    # Patch reaches back into the loaded module at import time, so this call
    # should patch the fresh class too.
    importlib.reload(cellflow_probes)
    cellflow_probes.patch_velocity_field()
    fresh_cls_after = mod.ConditionalVelocityField
    assert getattr(fresh_cls_after, "__cellflow_probed__", False) is True


def test_patched_otfm_forward_bit_equal_to_original():
    vf = _build_otfm_vf()
    rng, t, x, cond, encoder_noise = _init_inputs_otfm()
    variables = vf.init(rng, t, x, cond, encoder_noise, train=False)

    unpatched_out, unpatched_mean, unpatched_logvar = vf.apply(
        variables, t, x, cond, encoder_noise, train=False
    )

    patch_velocity_field()
    patched_out, patched_mean, patched_logvar = vf.apply(
        variables, t, x, cond, encoder_noise, train=False
    )

    assert jnp.array_equal(unpatched_out, patched_out)
    assert jnp.array_equal(unpatched_mean, patched_mean)
    assert jnp.array_equal(unpatched_logvar, patched_logvar)


def test_patched_genot_forward_bit_equal_to_original():
    vf = _build_genot_vf()
    rng, t, x, x_0, cond, encoder_noise = _init_inputs_genot()
    variables = vf.init(rng, t, x, x_0, cond, encoder_noise, train=False)

    unpatched_out, unpatched_mean, unpatched_logvar = vf.apply(
        variables, t, x, x_0, cond, encoder_noise, train=False
    )

    patch_velocity_field()
    patched_out, patched_mean, patched_logvar = vf.apply(
        variables, t, x, x_0, cond, encoder_noise, train=False
    )

    assert jnp.array_equal(unpatched_out, patched_out)
    assert jnp.array_equal(unpatched_mean, patched_mean)
    assert jnp.array_equal(unpatched_logvar, patched_logvar)


def test_monkey_patch_signature_parity():
    for cls in (ConditionalVelocityField, GENOTConditionalVelocityField):
        before = inspect.signature(cls.__call__)
        original_call = cls.__call__
        patch_velocity_field()
        after = inspect.signature(cls.__call__)
        unpatch_velocity_field()
        # Compare full inspect.Parameter per slot (name, kind, default) —
        # names-only equality would miss a `train=True → train=False`
        # default drift or a POSITIONAL_OR_KEYWORD → KEYWORD_ONLY change
        # that silently breaks positional callers inside cellflow.
        before_params = list(before.parameters.values())
        after_params = list(after.parameters.values())
        assert len(before_params) == len(after_params)
        for b, a in zip(before_params, after_params):
            assert b.name == a.name
            assert b.kind == a.kind
            assert b.default == a.default
        assert cls.__call__ is original_call


def test_otfm_sow_keys_are_exactly_expected():
    vf = _build_otfm_vf()
    rng, t, x, cond, encoder_noise = _init_inputs_otfm()
    variables = vf.init(rng, t, x, cond, encoder_noise, train=False)

    patch_velocity_field()
    _, mutated = vf.apply(
        variables, t, x, cond, encoder_noise, train=False,
        mutable=["intermediates"],
    )
    inters = mutated["intermediates"]
    expected = {
        "time_enc", "x_enc", "condition_mean",
        "pre_decoder", "decoder", "output",
    }
    assert set(inters.keys()) == expected
    # sow stores each name as a tuple of values; exactly one sow per name
    # per forward. A duplicate sow (e.g. post-refactor copy-paste bug) would
    # grow the tuple and fall through the set-equality above.
    for name in expected:
        assert len(inters[name]) == 1, f"sow {name!r} fired {len(inters[name])} times"


def test_genot_sow_keys_include_x_0_enc():
    vf = _build_genot_vf()
    rng, t, x, x_0, cond, encoder_noise = _init_inputs_genot()
    variables = vf.init(rng, t, x, x_0, cond, encoder_noise, train=False)

    patch_velocity_field()
    _, mutated = vf.apply(
        variables, t, x, x_0, cond, encoder_noise, train=False,
        mutable=["intermediates"],
    )
    inters = mutated["intermediates"]
    expected = {
        "time_enc", "x_enc", "x_0_enc", "condition_mean",
        "pre_decoder", "decoder", "output",
    }
    assert set(inters.keys()) == expected
    for name in expected:
        assert len(inters[name]) == 1, f"sow {name!r} fired {len(inters[name])} times"
