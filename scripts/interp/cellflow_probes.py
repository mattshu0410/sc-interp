from __future__ import annotations

import jax.numpy as jnp

_PROBED_ATTR = "__cellflow_probed__"
_ORIGINAL_CALL_ATTR = "_cellflow_original_call"


def _patched_call_otfm(
    self,
    t,
    x_t,
    cond,
    encoder_noise,
    train: bool = True,
):
    from cellflow.networks._utils import sinusoidal_time_encoder

    squeeze = x_t.ndim == 1
    cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
    if self.condition_mode == "deterministic":
        cond_embedding = cond_mean
    else:
        cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)
    cond_embedding = self.layer_cond_output_dropout(
        cond_embedding, deterministic=not train
    )

    t_encoded = sinusoidal_time_encoder(
        t, time_freqs=self.time_freqs, time_max_period=self.time_max_period
    )
    t_encoded = self.time_encoder(t_encoded, training=train)
    x_encoded = self.x_encoder(x_t, training=train)

    t_encoded = self.layer_norm_time(t_encoded)
    x_encoded = self.layer_norm_x(x_encoded)
    cond_embedding = self.layer_norm_condition(cond_embedding)

    if squeeze:
        cond_embedding = jnp.squeeze(cond_embedding)
    elif cond_embedding.shape[0] != x_t.shape[0]:
        cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

    self.sow("intermediates", "time_enc", t_encoded)
    self.sow("intermediates", "x_enc", x_encoded)
    # Post-tile, layout "BD". In deterministic mode (default) this equals
    # broadcast(cond_mean); in stochastic mode it includes the reparam sample.
    self.sow("intermediates", "condition_mean", cond_embedding)

    if self.conditioning == "concatenation":
        out = jnp.concatenate((t_encoded, x_encoded, cond_embedding), axis=-1)
    elif self.conditioning == "film":
        out = self.film_block(
            x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1)
        )
    elif self.conditioning == "resnet":
        out = self.resnet_block(
            x_encoded, jnp.concatenate((t_encoded, cond_embedding), axis=-1)
        )
    else:
        raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")
    self.sow("intermediates", "pre_decoder", out)

    out = self.decoder(out, training=train)
    self.sow("intermediates", "decoder", out)

    output = self.output_layer(out)
    self.sow("intermediates", "output", output)
    return output, cond_mean, cond_logvar


def _patched_call_genot(
    self,
    t,
    x_t,
    x_0,
    cond,
    encoder_noise,
    train: bool = True,
):
    from cellflow.networks._utils import sinusoidal_time_encoder

    squeeze = x_t.ndim == 1
    cond_mean, cond_logvar = self.condition_encoder(cond, training=train)
    if self.condition_mode == "deterministic":
        cond_embedding = cond_mean
    else:
        cond_embedding = cond_mean + encoder_noise * jnp.exp(cond_logvar / 2.0)
    cond_embedding = self.layer_cond_output_dropout(
        cond_embedding, deterministic=not train
    )
    t_encoded = sinusoidal_time_encoder(
        t, time_freqs=self.time_freqs, time_max_period=self.time_max_period
    )
    t_encoded = self.time_encoder(t_encoded, training=train)
    x_encoded = self.x_encoder(x_t, training=train)
    x_0_encoded = self.x_0_encoder(x_0, training=train)

    t_encoded = self.layer_norm_time(t_encoded)
    x_encoded = self.layer_norm_x(x_encoded)
    x_0_encoded = self.layer_norm_x_0(x_0_encoded)
    cond_embedding = self.layer_norm_condition(cond_embedding)

    if squeeze:
        cond_embedding = jnp.squeeze(cond_embedding)
    elif cond_embedding.shape[0] != x_t.shape[0]:
        cond_embedding = jnp.tile(cond_embedding, (x_t.shape[0], 1))

    self.sow("intermediates", "time_enc", t_encoded)
    self.sow("intermediates", "x_enc", x_encoded)
    self.sow("intermediates", "x_0_enc", x_0_encoded)
    self.sow("intermediates", "condition_mean", cond_embedding)

    if self.conditioning == "concatenation":
        out = jnp.concatenate(
            (t_encoded, x_encoded, x_0_encoded, cond_embedding), axis=-1
        )
    elif self.conditioning == "film":
        out = self.film_block(
            x_encoded,
            jnp.concatenate((t_encoded, x_0_encoded, cond_embedding), axis=-1),
        )
    elif self.conditioning == "resnet":
        out = self.resnet_block(
            x_encoded,
            jnp.concatenate((t_encoded, x_0_encoded, cond_embedding), axis=-1),
        )
    else:
        raise ValueError(f"Unknown conditioning mode: {self.conditioning}.")
    self.sow("intermediates", "pre_decoder", out)

    out = self.decoder(out, training=train)
    self.sow("intermediates", "decoder", out)

    output = self.output_layer(out)
    self.sow("intermediates", "output", output)
    return output, cond_mean, cond_logvar


def patch_velocity_field() -> None:
    from cellflow.networks._velocity_field import (
        ConditionalVelocityField,
        GENOTConditionalVelocityField,
    )

    # Guard against double-patch. importlib.reload creates a fresh class
    # without the attr, so the reload path naturally re-patches.
    for cls, patched in (
        (ConditionalVelocityField, _patched_call_otfm),
        (GENOTConditionalVelocityField, _patched_call_genot),
    ):
        if getattr(cls, _PROBED_ATTR, False):
            continue
        setattr(cls, _ORIGINAL_CALL_ATTR, cls.__call__)
        cls.__call__ = patched
        setattr(cls, _PROBED_ATTR, True)


def unpatch_velocity_field() -> None:
    from cellflow.networks._velocity_field import (
        ConditionalVelocityField,
        GENOTConditionalVelocityField,
    )

    for cls in (ConditionalVelocityField, GENOTConditionalVelocityField):
        if not getattr(cls, _PROBED_ATTR, False):
            continue
        cls.__call__ = getattr(cls, _ORIGINAL_CALL_ATTR)
        delattr(cls, _ORIGINAL_CALL_ATTR)
        delattr(cls, _PROBED_ATTR)
