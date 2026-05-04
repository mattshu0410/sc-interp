"""DiffPair → tensor batch streams for crosscoder training and scoring."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from scripts.diffing.base import DiffPair
from scripts.interp.hook_readers import H5ActivationReader


def get_activation_dim(pair: DiffPair) -> int:
    fs_a = pair.a.feature_shape()
    fs_b = pair.b.feature_shape()
    if fs_a != fs_b:
        raise ValueError(
            f"crosscoder requires matching feature shapes, got a={fs_a} b={fs_b}"
        )
    if len(fs_a) not in (1, 2):
        raise ValueError(f"crosscoder: unsupported feature shape {fs_a}")
    return fs_a[-1]


def _expand_to_samples(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if a.ndim == 3:
        n, t, d = a.shape
        return a.reshape(-1, d), b.reshape(-1, d), t
    if a.ndim == 2:
        return a, b, 1
    raise ValueError(f"crosscoder dataloader: unsupported chunk ndim {a.ndim}")


def normalizer_for_pair(pair: DiffPair) -> tuple[torch.Tensor, torch.Tensor]:
    """Read both sides' running_stats and return (mean, std) of shape (2, D).

    Stats must already be per-feature-dim. Raises FileNotFoundError if a
    side lacks the sidecar (legacy capture) or ValueError if the stored
    shape isn't 1-D — caller should run scripts.interp.backfill_running_stats.
    """
    means: list[torch.Tensor] = []
    stds: list[torch.Tensor] = []
    for src, side in [(pair.a, "a"), (pair.b, "b")]:
        with H5ActivationReader(src.path) as r:
            stats = r.running_stats(src.capture, src.tags)
        if stats is None:
            raise FileNotFoundError(
                f"running_stats missing for pair side {side!r} "
                f"(capture={src.capture!r}, tags={src.tags!r}) at {src.path}. "
                f"Run: python -m scripts.interp.backfill_running_stats {src.path}"
            )
        if stats.mean.ndim != 1:
            raise ValueError(
                f"running_stats for side {side!r} have shape "
                f"{tuple(stats.mean.shape)}; crosscoder needs per-D stats. "
                f"Re-run scripts.interp.backfill_running_stats {src.path} "
                f"to pool the token axis into samples."
            )
        means.append(stats.mean)
        stds.append(stats.std(unbiased=False))
    return torch.stack(means, dim=0), torch.stack(stds, dim=0)


def iter_pair_samples(
    pair: DiffPair,
    *,
    batch_size: int,
    chunk_rows: int,
    device: str | torch.device,
    shuffle: bool = True,
) -> Iterator[tuple[torch.Tensor, dict[str, np.ndarray], int]]:
    """One epoch over the pair, yielding (batch, batch_labels, tokens_per_cell).

    Batch shape: (batch_size, 2, D). For BTD captures each cell expands
    into T samples and per-cell labels are repeated T times.
    """
    a_iter = pair.a.iter_chunks(chunk_rows)
    b_iter = pair.b.iter_chunks(chunk_rows)
    for (a_chunk, a_labels), (b_chunk, _) in zip(a_iter, b_iter):
        a_aligned, b_aligned = pair.alignment.apply(a_chunk, b_chunk)
        a_samples, b_samples, tokens_per_cell = _expand_to_samples(a_aligned, b_aligned)
        x = torch.stack([a_samples, b_samples], dim=1)
        sample_labels = (
            {k: np.repeat(v, tokens_per_cell, axis=0) for k, v in a_labels.items()}
            if tokens_per_cell > 1
            else dict(a_labels)
        )
        n = x.shape[0]
        if shuffle:
            perm = torch.randperm(n)
            x = x[perm]
            perm_np = perm.numpy()
            sample_labels = {k: v[perm_np] for k, v in sample_labels.items()}
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            yield (
                x[start:stop].to(device, non_blocking=True),
                {k: v[start:stop] for k, v in sample_labels.items()},
                tokens_per_cell,
            )


def infinite_pair_samples(pair: DiffPair, **kwargs) -> Iterator[torch.Tensor]:
    """Tensor-only stream that loops indefinitely (training)."""
    while True:
        for tensor, _labels, _tpc in iter_pair_samples(pair, **kwargs):
            yield tensor


def materialize_validation(
    pair: DiffPair,
    *,
    batch_size: int,
    chunk_rows: int,
    device: str | torch.device,
    n_batches: int,
) -> list[torch.Tensor]:
    """Sized list of validation batches (the trainer expects len() on it)."""
    out: list[torch.Tensor] = []
    for tensor, _labels, _tpc in iter_pair_samples(
        pair, batch_size=batch_size, chunk_rows=chunk_rows, device=device, shuffle=False,
    ):
        out.append(tensor)
        if len(out) >= n_batches:
            break
    return out
