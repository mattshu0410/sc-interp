"""Dictionary-specific feature selection and encoding.

The `FeatureSource` Protocol is the only place crosscoder/SAE knowledge
lives. Everything downstream — profiling, selection, annotation —
consumes raw `(B*T, dict_size)` codes and is dictionary-agnostic.

To support a new dictionary type (SAE, transcoder, etc.), implement a
new `FeatureSource` in this directory. No other module changes.
"""
from __future__ import annotations

from typing import Protocol

import torch


class FeatureSource(Protocol):
    """Selects ESM-specific features and encodes activations to codes.

    Adapter over a trained dictionary (crosscoder / SAE / ...). The
    walker reads only the sides this source declares in `needed_sides`
    from the cached activation H5s, then calls `encode` per chunk.
    """

    @property
    def dict_size(self) -> int:
        """Number of latent codes per token."""

    @property
    def needed_sides(self) -> list[str]:
        """Side keys the encoder needs, e.g. ``["a","b"]`` for a crosscoder
        or ``[""]`` for a single-condition SAE. Matches the keys in the
        dict passed to `encode`.
        """

    def select_features(self, k: int) -> list[int]:
        """Top-K most ESM-specific feature indices, sorted descending."""

    def encode(self, h_per_side: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode flat activation samples to codes.

        Inputs are ``(N, D)`` per side, where N is samples (cells × tokens
        flattened) and D is the per-token activation dim. Output is
        ``(N, dict_size)`` codes.
        """
