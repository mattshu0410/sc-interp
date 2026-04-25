"""Alignment strategies for bringing two activation spaces into comparison.

Kept in a separate module from methods because several methods reuse the
same fit (e.g., an L2 metric and a shared-SAE crosscoder both project
through the same Procrustes rotation). If alignment lived inside one
method, the next would re-derive it.

Only `Identity` is implemented in iteration 1 — that's what finetune-vs-
base same-arch diffing needs. `Procrustes`, `CCA`, and `SharedSAE` are
stubbed so the ABC and registry are stable; they raise NotImplementedError
until their iteration lands.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Alignment(ABC):
    name: str

    def fit(self, a: torch.Tensor, b: torch.Tensor) -> None:
        return None

    @abstractmethod
    def apply(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class Identity(Alignment):
    """Pass-through. For same-arch, same-layer pairs where activations are
    already in a directly comparable space — the dominant near-term case."""

    name = "identity"

    def apply(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if a.shape != b.shape:
            raise ValueError(
                f"Identity alignment requires matching shapes, got "
                f"a={tuple(a.shape)} b={tuple(b.shape)}"
            )
        return a, b


class Procrustes(Alignment):
    name = "procrustes"

    def apply(self, a, b):
        raise NotImplementedError(
            "Procrustes alignment lands in iteration 2 with cross-arch methods."
        )


class CCA(Alignment):
    name = "cca"

    def apply(self, a, b):
        raise NotImplementedError(
            "CCA alignment lands in iteration 3."
        )


class SharedSAE(Alignment):
    name = "shared_sae"

    def apply(self, a, b):
        raise NotImplementedError(
            "SharedSAE alignment lands with the crosscoder iteration."
        )


_ALIGNMENTS: dict[str, type[Alignment]] = {
    cls.name: cls for cls in (Identity, Procrustes, CCA, SharedSAE)
}


def get_alignment(name: str) -> Alignment:
    if name not in _ALIGNMENTS:
        raise KeyError(
            f"unknown alignment {name!r}. Available: {sorted(_ALIGNMENTS)}"
        )
    return _ALIGNMENTS[name]()
