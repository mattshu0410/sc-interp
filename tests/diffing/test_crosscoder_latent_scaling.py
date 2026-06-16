from __future__ import annotations

import pytest
import torch

pytest.importorskip("dictionary_learning")

from scripts.diffing.dictionary_learning_adapter import CrossCoder  # noqa: E402
from scripts.diffing.methods.crosscoder.latent_scaling import (  # noqa: E402
    LatentScalingResults,
    closed_form_scalars,
    identify_latent_sets,
    remove_latents,
)


# ----- closed_form_scalars math -----------------------------------------------


class _MockModel(torch.nn.Module):
    """Minimal duck-typed dict_model: has dict_size and an encode that returns a
    pre-baked latent activation tensor regardless of input."""

    def __init__(self, dict_size: int, latent_acts: torch.Tensor):
        super().__init__()
        self.dict_size = dict_size
        self._latent_acts = latent_acts

    def encode(self, batch, use_threshold: bool = True):
        n = batch.shape[0]
        return self._latent_acts[:n]


def test_closed_form_scalars_recovers_planted_beta_k1():
    # One latent, one batch: argmin_β ‖β · f · v - β_true · f · v‖² has a unique
    # global min at β = β_true.
    torch.manual_seed(0)
    n, d = 200, 6
    f = torch.rand(n, 1) + 0.1   # strictly positive so every sample contributes
    v = torch.zeros(1, d)
    v[0, 0] = 1.0
    beta_true = 2.7
    y = beta_true * f * v        # shape (n, d)

    def target_fn(batch, **_kwargs):
        return y

    model = _MockModel(dict_size=1, latent_acts=f)
    batches = [torch.zeros(n, 2, d)]  # arbitrary; encode ignores it

    betas, counts = closed_form_scalars(
        latent_vectors=v,
        latent_indices=torch.tensor([0]),
        batches=batches,
        dict_model=model,
        target_activation_fn=target_fn,
        device="cpu",
        progress=False,
    )
    assert betas.shape == (1,)
    assert counts.shape == (1,)
    assert counts[0].item() == n
    torch.testing.assert_close(betas[0], torch.tensor(beta_true), atol=1e-4, rtol=1e-4)


def test_closed_form_scalars_recovers_planted_beta_orthogonal_k2():
    # Two orthogonal latent vectors, two latents whose activations have disjoint
    # support across samples (so cross-correlation Σ f_1·f_2 = 0). Each latent's
    # regression should recover its own β_true independently.
    torch.manual_seed(1)
    n, d = 200, 6
    f = torch.zeros(n, 2)
    f[: n // 2, 0] = torch.rand(n // 2) + 0.1   # first half: only latent 0 fires
    f[n // 2 :, 1] = torch.rand(n // 2) + 0.1   # second half: only latent 1
    v = torch.zeros(2, d)
    v[0, 0] = 1.0
    v[1, 1] = 1.0
    beta_true = torch.tensor([2.0, -3.5])
    y = (f[:, 0:1] * beta_true[0]) * v[0:1] + (f[:, 1:2] * beta_true[1]) * v[1:2]

    def target_fn(batch, **_kwargs):
        return y

    model = _MockModel(dict_size=2, latent_acts=f)
    betas, counts = closed_form_scalars(
        latent_vectors=v,
        latent_indices=torch.tensor([0, 1]),
        batches=[torch.zeros(n, 2, d)],
        dict_model=model,
        target_activation_fn=target_fn,
        device="cpu",
        progress=False,
    )
    torch.testing.assert_close(betas, beta_true, atol=1e-4, rtol=1e-4)


def test_closed_form_scalars_streams_across_batches():
    # Splitting the same data into multiple batches must give the same betas as
    # one big batch (the accumulator is associative by construction).
    torch.manual_seed(2)
    n, d = 240, 4
    f = torch.rand(n, 1) + 0.1
    v = torch.zeros(1, d)
    v[0, 0] = 1.0
    beta_true = 1.3
    y = beta_true * f * v

    def target_fn(batch, **_kwargs):
        return y[target_fn.cursor : target_fn.cursor + batch.shape[0]]

    target_fn.cursor = 0  # type: ignore[attr-defined]

    # We need a more careful mock when batching: encode must hand back the right
    # rows of f for the current batch. Pre-shard f so each call slices in order.
    chunks = list(torch.split(torch.arange(n), 70))

    class _StreamMock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dict_size = 1
            self._cursor = 0

        def encode(self, batch, use_threshold: bool = True):
            k = batch.shape[0]
            out = f[self._cursor : self._cursor + k]
            self._cursor += k
            return out

    def target_fn_streaming(batch, **_kwargs):
        # Mirror the same cursor logic as the encode mock.
        target_fn_streaming.cursor += batch.shape[0]  # type: ignore[attr-defined]
        return y[target_fn_streaming.cursor - batch.shape[0] : target_fn_streaming.cursor]

    target_fn_streaming.cursor = 0  # type: ignore[attr-defined]

    batches = [torch.zeros(len(c), 2, d) for c in chunks]
    betas, _ = closed_form_scalars(
        latent_vectors=v,
        latent_indices=torch.tensor([0]),
        batches=batches,
        dict_model=_StreamMock(),
        target_activation_fn=target_fn_streaming,
        device="cpu",
        progress=False,
    )
    torch.testing.assert_close(betas[0], torch.tensor(beta_true), atol=1e-4, rtol=1e-4)


# ----- remove_latents ---------------------------------------------------------


def test_remove_latents_shape():
    n, d, k = 5, 4, 3
    activation = torch.randn(n, d)
    latent_acts = torch.randn(n, k)
    latent_vecs = torch.randn(k, d)
    out = remove_latents(activation, latent_acts, latent_vecs)
    assert out.shape == (k, n, d)


def test_remove_latents_subtracts_correctly():
    # For each latent j: out[j] = activation - f_j * v_j (broadcasted across rows).
    activation = torch.tensor([[1.0, 2.0, 3.0]])
    latent_acts = torch.tensor([[2.0, 3.0]])  # 1 row, 2 latents
    latent_vecs = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    out = remove_latents(activation, latent_acts, latent_vecs)
    expected_0 = activation - 2.0 * latent_vecs[0:1]  # [[ -1, 2, 3 ]]
    expected_1 = activation - 3.0 * latent_vecs[1:2]  # [[ 1, -1, 3 ]]
    torch.testing.assert_close(out[0], expected_0)
    torch.testing.assert_close(out[1], expected_1)


# ----- identify_latent_sets ---------------------------------------------------


def _construct_partitioned_crosscoder(d: int = 8, dict_size: int = 8) -> CrossCoder:
    """Build a CrossCoder whose decoder columns are deliberately partitioned:
    half a-specific (d_b ≈ 0), half b-specific (d_a ≈ 0), one shared, one shared."""
    model = CrossCoder(activation_dim=d, dict_size=dict_size, num_layers=2)
    with torch.no_grad():
        # decoder.weight: (num_layers=2, dict_size, activation_dim)
        model.decoder.weight.zero_()
        # latents 0..1: a-specific (d_a non-zero, d_b zero)
        model.decoder.weight[0, 0, 0] = 1.0
        model.decoder.weight[0, 1, 1] = 1.0
        # latents 2..3: b-specific (d_a zero, d_b non-zero)
        model.decoder.weight[1, 2, 2] = 1.0
        model.decoder.weight[1, 3, 3] = 1.0
        # latents 4..5: shared (d_a == d_b)
        model.decoder.weight[0, 4, 4] = 1.0
        model.decoder.weight[1, 4, 4] = 1.0
        model.decoder.weight[0, 5, 5] = 1.0
        model.decoder.weight[1, 5, 5] = 1.0
        # latents 6..7: very small random both sides — won't pass any threshold
    return model


def test_identify_latent_sets_partitions_correctly():
    model = _construct_partitioned_crosscoder()
    sets = identify_latent_sets(
        model,
        threshold_specific=0.9,
        threshold_shared_low=0.4,
        threshold_shared_high=0.6,
        n_shared_baseline=10,
    )
    a_idx = sets["a_specific"].tolist()
    b_idx = sets["b_specific"].tolist()
    shared_idx = sorted(sets["shared"].tolist())

    assert sorted(a_idx) == [0, 1], a_idx
    assert sorted(b_idx) == [2, 3], b_idx
    assert sorted([i for i in shared_idx if i in {4, 5}]) == [4, 5]


def test_identify_latent_sets_caps_shared_at_n_baseline():
    # Construct many shared latents and verify n_shared_baseline limits the sample.
    d = 16
    dict_size = 50
    model = CrossCoder(activation_dim=d, dict_size=dict_size, num_layers=2)
    with torch.no_grad():
        model.decoder.weight.zero_()
        for j in range(dict_size):
            model.decoder.weight[0, j, j % d] = 1.0
            model.decoder.weight[1, j, j % d] = 1.0   # all shared
    sets = identify_latent_sets(
        model,
        threshold_specific=0.9,
        threshold_shared_low=0.4,
        threshold_shared_high=0.6,
        n_shared_baseline=12,
    )
    assert len(sets["a_specific"]) == 0
    assert len(sets["b_specific"]) == 0
    assert len(sets["shared"]) == 12


# ----- LatentScalingResults derived metrics -----------------------------------


def test_latent_scaling_results_nu_ratios_and_mask():
    # Hand-craft betas to verify ν computation and threshold filtering.
    indices = torch.tensor([0, 1, 2, 3])
    r = LatentScalingResults(indices=indices, side="a")
    # Owner = a, Other = b. Truly-specific should pass both thresholds.
    r.betas = {
        "a_recon": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "b_recon": torch.tensor([0.05, 0.6, 0.05, 0.6]),    # <0.5 for indices 0, 2
        "a_error": torch.tensor([1.0, 1.0, 1.0, 1.0]),
        "b_error": torch.tensor([0.05, 0.05, 0.4, 0.4]),    # <0.2 for indices 0, 1
    }
    nu_r = r.nu_r()
    nu_e = r.nu_e()
    torch.testing.assert_close(nu_r, torch.tensor([0.05, 0.6, 0.05, 0.6]))
    torch.testing.assert_close(nu_e, torch.tensor([0.05, 0.05, 0.4, 0.4]))

    mask = r.truly_specific_mask(nu_r_threshold=0.5, nu_e_threshold=0.2)
    assert mask.tolist() == [True, False, False, False]   # only index 0 passes both
