"""Latent Scaling validity diagnostics for crosscoder model-specific latents.

`closed_form_scalars` and `remove_latents` are vendored from
`science-of-finetuning/diffing-toolkit` (latent_scaling/{closed_form.py, utils.py}).
The math implements eq. 4 of Minder et al., "Overcoming Sparsity Artifacts in
Crosscoders to Interpret Chat-Tuning" (NeurIPS 2025, arXiv:2504.02922).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import islice
from typing import Callable, Iterable

import torch
from tqdm import tqdm

from scripts.diffing.base import DiffPair
from scripts.diffing.dictionary_learning_adapter import (
    BatchTopKCrossCoder,
    CrossCoder,
)
from scripts.diffing.methods.crosscoder.dataloader import iter_pair_samples


# =============================================================================
# Vendored from science-of-finetuning/diffing-toolkit
# =============================================================================


def remove_latents(
    activation: torch.Tensor,
    latent_activations: torch.Tensor,
    latent_vectors: torch.Tensor,
) -> torch.Tensor:
    """Per-latent activation with each latent's contribution subtracted.

    activation: (N, D); latent_activations: (N, K); latent_vectors: (K, D).
    Returns (K, N, D) — for each of the K latents, the activation with that
    latent's own decoder contribution removed.
    """
    K, D = latent_vectors.shape
    N = activation.shape[0]
    activation_stacked = activation.unsqueeze(0).expand(K, N, D)
    contribution = latent_vectors.unsqueeze(1) * latent_activations.T.unsqueeze(-1)
    return activation_stacked - contribution


@torch.no_grad()
def closed_form_scalars(
    latent_vectors: torch.Tensor,
    latent_indices: torch.Tensor,
    batches: Iterable[torch.Tensor],
    dict_model: CrossCoder | BatchTopKCrossCoder,
    target_activation_fn: Callable[..., torch.Tensor],
    *,
    device: torch.device | str = "cuda",
    dtype: torch.dtype = torch.float32,
    progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Closed-form solution to argmin_β ‖β·f·d - y‖² for each (latent, target) pair.

    For latent j with decoder vector dⱼ and per-input activations fⱼ(xᵢ) and target yᵢ:
        β_j = ⟨dⱼ, Σᵢ fⱼ(xᵢ) yᵢ⟩ / (‖dⱼ‖² · Σᵢ fⱼ(xᵢ)²)

    Streams `batches` once. target_activation_fn returns Y of shape (N, D) for
    same-target-for-all-latents, or (K, N, D) for per-latent targets (e.g. error
    with self-removed).
    """
    assert latent_vectors.ndim == 2
    latent_vectors = latent_vectors.to(device).to(dtype)
    latent_indices = latent_indices.to(device)
    K, D = latent_vectors.shape
    dict_size = dict_model.dict_size

    A = torch.zeros(D, K, device=device, dtype=dtype)
    C = torch.zeros(K, device=device, dtype=dtype)
    count_active = torch.zeros(K, device=device, dtype=dtype)

    iterator = tqdm(batches, desc="latent_scaling") if progress else batches
    for batch in iterator:
        if batch.shape[0] == 0:
            continue
        batch = batch.to(device).to(dtype)
        latent_activations = dict_model.encode(batch, use_threshold=True)
        assert latent_activations.shape == (batch.shape[0], dict_size)

        Y_batch = target_activation_fn(
            batch,
            crosscoder=dict_model,
            latent_activations=latent_activations,
            latent_indices=latent_indices,
            latent_vectors=latent_vectors,
        )

        latent_activations = latent_activations[:, latent_indices]
        non_zero_per_latent = (latent_activations != 0).sum(dim=0)
        count_active += non_zero_per_latent
        non_zero_mask = non_zero_per_latent > 0

        if Y_batch.dim() == 3:
            assert Y_batch.shape == (K, batch.shape[0], D)
            A_update = (
                torch.matmul(
                    Y_batch[non_zero_mask].transpose(1, 2),
                    latent_activations[:, non_zero_mask].T.unsqueeze(-1),
                )
                .squeeze(-1)
                .T
            )
        else:
            assert Y_batch.shape == (batch.shape[0], D)
            A_update = Y_batch.T @ latent_activations[:, non_zero_mask]

        A[:, non_zero_mask] += A_update
        C[non_zero_mask] += (latent_activations[:, non_zero_mask] ** 2).sum(dim=0)

    decoder_norms_sq = (latent_vectors ** 2).sum(dim=1)
    inner = torch.sum(A.T * latent_vectors, dim=1)
    betas = inner / (C * decoder_norms_sq).clamp_min(1e-30)
    return betas.cpu(), count_active.cpu()


# =============================================================================
# Crosscoder-specific target loaders (parameterised by layer)
# =============================================================================


def _layer_reconstruction_target(layer: int) -> Callable[..., torch.Tensor]:
    def fn(batch, *, crosscoder, latent_activations, **_kwargs):
        recon = crosscoder.decode(latent_activations, denormalize_activations=False)
        return recon[:, layer, :]
    return fn


def _layer_error_target(layer: int) -> Callable[..., torch.Tensor]:
    # Layer activation minus reconstruction-with-this-latent-removed (per-latent),
    # both in the model's normalized frame (decoder output is normalized via
    # denormalize_activations=False; batch needs explicit normalization to match).
    def fn(batch, *, crosscoder, latent_activations, latent_indices, **_kwargs):
        normalized_batch = crosscoder.normalize_activations(batch, inplace=False)
        layer_decoder_for_indices = crosscoder.decoder.weight[layer, latent_indices, :]
        recon = crosscoder.decode(latent_activations, denormalize_activations=False)
        return normalized_batch[:, layer, :] - remove_latents(
            recon[:, layer, :],
            latent_activations[:, latent_indices],
            layer_decoder_for_indices,
        )
    return fn


# =============================================================================
# Result type + orchestrator
# =============================================================================


@dataclass
class LatentScalingResults:
    """Per-latent betas and validity ratios for one latent set.

    `side` is "a" or "b" — the side these latents are claimed-specific to (the
    "owner" side in paper terminology). νʳ and νᵉ are computed as
    other-side / owner-side; values near 0 = truly specific.
    """
    indices: torch.Tensor
    side: str
    betas: dict[str, torch.Tensor] = field(default_factory=dict)
    counts: dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def _other_side(self) -> str:
        return "b" if self.side == "a" else "a"

    def nu_r(self) -> torch.Tensor:
        owner = self.betas[f"{self.side}_recon"]
        other = self.betas[f"{self._other_side}_recon"]
        return other / owner.abs().clamp_min(1e-12)

    def nu_e(self) -> torch.Tensor:
        owner = self.betas[f"{self.side}_error"]
        other = self.betas[f"{self._other_side}_error"]
        return other / owner.abs().clamp_min(1e-12)

    def truly_specific_mask(
        self, nu_r_threshold: float = 0.5, nu_e_threshold: float = 0.2,
    ) -> torch.Tensor:
        return (self.nu_r() < nu_r_threshold) & (self.nu_e() < nu_e_threshold)


def identify_latent_sets(
    model: CrossCoder | BatchTopKCrossCoder,
    *,
    threshold_specific: float,
    threshold_shared_low: float,
    threshold_shared_high: float,
    n_shared_baseline: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Returns indices for 'a_specific', 'b_specific', 'shared'.

    Specificity uses dec_norm_diff (the canonical per-feature ratio in [0, 1]).
    """
    norms = model.decoder.weight.detach().norm(dim=-1).cpu().float()
    n_a, n_b = norms[0], norms[1]
    diff_a = 0.5 * ((n_a - n_b) / torch.maximum(n_a, n_b).clamp_min(1e-12) + 1.0)
    diff_b = 1.0 - diff_a

    a_specific = torch.where(diff_a > threshold_specific)[0]
    b_specific = torch.where(diff_b > threshold_specific)[0]
    shared_pool = torch.where(
        (diff_b > threshold_shared_low) & (diff_b < threshold_shared_high)
    )[0]
    if len(shared_pool) <= n_shared_baseline:
        shared = shared_pool
    else:
        rng = torch.Generator().manual_seed(seed)
        shared = shared_pool[torch.randperm(len(shared_pool), generator=rng)[:n_shared_baseline]]
    return {"a_specific": a_specific, "b_specific": b_specific, "shared": shared}


def _tensor_only(it):
    for tensor, _labels, _tpc in it:
        yield tensor


def _compute_betas_for_set(
    model: CrossCoder | BatchTopKCrossCoder,
    pair: DiffPair,
    indices: torch.Tensor,
    side: str,
    *,
    batch_size: int,
    chunk_rows: int,
    device: str,
    progress: bool,
    num_samples: int | None = None,
) -> LatentScalingResults:
    layer_for_side = 0 if side == "a" else 1
    latent_vectors = model.decoder.weight[layer_for_side, indices, :].detach().to(device)
    latent_indices = indices.to(device)

    # The four targets feeding νʳ (recon) and νᵉ (error).
    targets = {
        "a_recon": _layer_reconstruction_target(0),
        "b_recon": _layer_reconstruction_target(1),
        "a_error": _layer_error_target(0),
        "b_error": _layer_error_target(1),
    }

    n_batches = (
        math.ceil(num_samples / batch_size) if num_samples is not None else None
    )

    result = LatentScalingResults(indices=indices.cpu(), side=side)
    for name, target_fn in targets.items():
        batches = _tensor_only(iter_pair_samples(
            pair, batch_size=batch_size, chunk_rows=chunk_rows,
            device=device, shuffle=False,
        ))
        if n_batches is not None:
            batches = islice(batches, n_batches)
        betas, counts = closed_form_scalars(
            latent_vectors=latent_vectors,
            latent_indices=latent_indices,
            batches=batches,
            dict_model=model,
            target_activation_fn=target_fn,
            device=device,
            progress=progress,
        )
        result.betas[name] = betas
        result.counts[name] = counts
    return result


def compute_latent_scaling(
    model: CrossCoder | BatchTopKCrossCoder,
    pair: DiffPair,
    *,
    threshold_specific: float = 0.9,
    threshold_shared_low: float = 0.4,
    threshold_shared_high: float = 0.6,
    n_shared_baseline: int = 100,
    batch_size: int = 1024,
    chunk_rows: int = 128,
    num_samples: int | None = 100_000,
    device: str | None = None,
    progress: bool = True,
) -> dict[str, LatentScalingResults]:
    """Run latent scaling on a trained crosscoder.

    Identifies a-specific, b-specific, and shared-baseline latent sets via
    dec_norm_diff thresholds, then runs 4 regressions (recon × 2 layers,
    error × 2 layers) per set. Returns a dict keyed by set name.
    """
    if model is None:
        raise ValueError("compute_latent_scaling requires a fitted crosscoder model")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    sets = identify_latent_sets(
        model,
        threshold_specific=threshold_specific,
        threshold_shared_low=threshold_shared_low,
        threshold_shared_high=threshold_shared_high,
        n_shared_baseline=n_shared_baseline,
    )

    out: dict[str, LatentScalingResults] = {}
    for set_name, indices in sets.items():
        if len(indices) == 0:
            out[set_name] = LatentScalingResults(
                indices=torch.empty(0, dtype=torch.long), side=set_name[0],
            )
            continue
        # For specific sets, the "owner" side is encoded in the set name.
        # For shared baselines, choose side='a' arbitrarily (ν is symmetric in
        # expectation for shared latents).
        side = "a" if set_name in ("a_specific", "shared") else "b"
        out[set_name] = _compute_betas_for_set(
            model, pair, indices, side,
            batch_size=batch_size, chunk_rows=chunk_rows,
            device=device, progress=progress, num_samples=num_samples,
        )
    return out


# =============================================================================
# Persistence + CLI
# =============================================================================


_LS_FILENAME = "latent_scaling.h5"


def write_results(results: dict[str, LatentScalingResults], out_path) -> None:
    import h5py
    import numpy as np
    from pathlib import Path

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for set_name, r in results.items():
            grp = f.create_group(set_name)
            grp.attrs["side"] = r.side
            grp.create_dataset("indices", data=r.indices.numpy().astype(np.int64))
            betas_grp = grp.create_group("betas")
            counts_grp = grp.create_group("counts")
            for name, t in r.betas.items():
                betas_grp.create_dataset(name, data=t.numpy().astype(np.float32))
            for name, t in r.counts.items():
                counts_grp.create_dataset(name, data=t.numpy().astype(np.float32))
            if len(r.indices) > 0:
                grp.create_dataset("nu_r", data=r.nu_r().numpy().astype(np.float32))
                grp.create_dataset("nu_e", data=r.nu_e().numpy().astype(np.float32))
                grp.create_dataset(
                    "truly_specific",
                    data=r.truly_specific_mask().numpy().astype(np.uint8),
                )


def _summarise(results: dict[str, LatentScalingResults]) -> str:
    lines = []
    for set_name, r in results.items():
        n = int(r.indices.shape[0])
        lines.append(f"{set_name}: {n} latents")
        if n == 0:
            continue
        truly = int(r.truly_specific_mask().sum())
        nu_r_med = float(r.nu_r().median())
        nu_e_med = float(r.nu_e().median())
        lines.append(
            f"  truly_specific (νʳ<0.5 ∧ νᵉ<0.2): {truly}/{n} ({100 * truly / n:.1f}%)  "
            f"median νʳ={nu_r_med:.3f}  median νᵉ={nu_e_med:.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    from omegaconf import OmegaConf

    from scripts.diffing.methods.crosscoder.method import CrossCoderMethod
    from scripts.diffing.pairs import load_pair
    from scripts.interp.hook_readers import H5ActivationReader

    p = argparse.ArgumentParser(
        prog="python -m scripts.diffing.methods.crosscoder.latent_scaling",
        description="Run Latent Scaling diagnostics on a trained crosscoder.",
    )
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--device", default=None)
    p.add_argument("--n-shared-baseline", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--chunk-rows", type=int, default=128)
    p.add_argument("--threshold-specific", type=float, default=0.9)
    p.add_argument("--num-samples", type=int, default=100_000)
    args = p.parse_args(argv)

    scores_path = args.model_dir / "scores.h5"
    with H5ActivationReader(scores_path) as r:
        meta = r.meta
    cfg = OmegaConf.create(str(meta["config"]))
    pair = load_pair(
        meta["pair_a_path"],
        meta["pair_b_path"],
        capture=meta["pair_capture"],
        alignment=meta.get("alignment", "identity"),
        relationship=meta.get("relationship", "unspecified"),
        tags_a=dict(cfg.pair.a.get("tags", {})),
        tags_b=dict(cfg.pair.b.get("tags", {})),
    )

    method = CrossCoderMethod.load(args.model_dir)
    results = compute_latent_scaling(
        method.model,
        pair,
        threshold_specific=args.threshold_specific,
        n_shared_baseline=args.n_shared_baseline,
        batch_size=args.batch_size,
        chunk_rows=args.chunk_rows,
        num_samples=(args.num_samples if args.num_samples > 0 else None),
        device=args.device,
    )

    out_path = args.model_dir / _LS_FILENAME
    write_results(results, out_path)

    print(_summarise(results))
    print(f"==> wrote {out_path}")


if __name__ == "__main__":
    main()
