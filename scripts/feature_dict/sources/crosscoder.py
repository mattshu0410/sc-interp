"""Crosscoder-backed FeatureSource.

Selects ESM-specific features by ``feature_dec_norm_diff_b`` (Minder
2025 §3.1: a valid proxy for BatchTopK crosscoders, ν correlation
≥ 0.73). Encodes ``(h_base, h_esm)`` activation pairs into shared codes
via the trained dictionary's own ``encode``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.diffing.methods.crosscoder.method import CrossCoderMethod


@dataclass
class CrosscoderSource:
    """FeatureSource over a trained `CrossCoderMethod` artifact.

    Args:
        model_dir: Directory containing ``model_final.pt`` + ``config.json``
            (output of ``CrossCoderMethod.save``).
        side_a: Tag for the base-side activation key (default ``"a"``).
        side_b: Tag for the ESM-side activation key (default ``"b"``).
        device: Where to run encode (``"cuda"`` or ``"cpu"``).
    """
    model_dir: Path
    side_a: str = "a"
    side_b: str = "b"
    device: str = "cuda"

    @cached_property
    def _method(self) -> CrossCoderMethod:
        return CrossCoderMethod.load(Path(self.model_dir))

    @cached_property
    def _model(self) -> torch.nn.Module:
        m = self._method.model
        if m is None:
            raise RuntimeError(f"crosscoder at {self.model_dir} did not load")
        m.eval()
        m.to(self.device)
        return m

    @cached_property
    def _scores(self) -> dict[str, np.ndarray]:
        scores_path = self.model_dir / "scores.h5"
        if not scores_path.exists():
            raise FileNotFoundError(
                f"missing {scores_path}; run the diffing CLI's score step "
                f"to populate per-feature norms"
            )
        with h5py.File(scores_path) as f:
            return {
                "feature_norm_a": f["feature_norm_a/activation"][0].astype(np.float32),
                "feature_norm_b": f["feature_norm_b/activation"][0].astype(np.float32),
                "dec_norm_diff_a": f["feature_dec_norm_diff_a/activation"][0].astype(np.float32),
                "dec_norm_diff_b": f["feature_dec_norm_diff_b/activation"][0].astype(np.float32),
            }

    @property
    def dict_size(self) -> int:
        return int(self._scores["dec_norm_diff_b"].shape[0])

    @property
    def needed_sides(self) -> list[str]:
        return [self.side_a, self.side_b]

    @property
    def scores(self) -> dict[str, np.ndarray]:
        """Per-feature score arrays read from ``scores.h5``. Useful for
        the cheap index pass."""
        return self._scores

    def select_features(self, k: int) -> list[int]:
        """Top-K features by ``feature_dec_norm_diff_b``, descending."""
        if k > self.dict_size:
            raise ValueError(f"k={k} > dict_size={self.dict_size}")
        order = np.argsort(-self._scores["dec_norm_diff_b"])
        return order[:k].astype(np.int64).tolist()

    @torch.no_grad()
    def encode(self, h_per_side: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode flat ``(N, D)`` per-side activations to ``(N, dict_size)``.

        The trained model handles its own activation normalization; pass
        raw (un-normalized) activations matching what was captured to H5.
        """
        h_a = h_per_side[self.side_a]
        h_b = h_per_side[self.side_b]
        if h_a.shape != h_b.shape:
            raise ValueError(f"side shape mismatch: a={h_a.shape} b={h_b.shape}")
        # (N, 2, D) is the layout `iter_pair_samples` yields and what
        # `BatchTopKCrossCoder.encode` consumes.
        x = torch.stack([h_a, h_b], dim=1).to(self.device, dtype=torch.float32)
        codes = self._model.encode(x)
        return codes
