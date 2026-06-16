"""Streaming per-feature aggregators.

Two levels in one pass:

* :class:`ScalarStats` — cheap scalars for all features (mean, max,
  frac_active). Powers the cross-feature index.
* :class:`RichStats` — per-tracked-feature gene aggregation, per-cell-line
  and per-pert marginals, top-N cells. Powers the deep cards.

The :class:`Aggregator` consumes ``Chunk`` objects from any walker
(crosscoder or future SAE) and is dictionary-agnostic.
"""
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch

from scripts.feature_dict.walker import Chunk


# ── Scalar stats: all features ──────────────────────────────────────────────


@dataclass
class ScalarStats:
    """Per-feature running scalars across all chunks.

    Storage is ``(F,)`` arrays kept on CPU as float64 to avoid drift over
    millions of token additions. ``finalize`` collapses to a dict of
    summary stats keyed by name.
    """
    n_features: int
    sum: np.ndarray = field(init=False)
    sumsq: np.ndarray = field(init=False)
    max: np.ndarray = field(init=False)
    active_count: np.ndarray = field(init=False)
    total_count: int = 0

    def __post_init__(self) -> None:
        self.sum = np.zeros(self.n_features, dtype=np.float64)
        self.sumsq = np.zeros(self.n_features, dtype=np.float64)
        self.max = np.zeros(self.n_features, dtype=np.float32)
        self.active_count = np.zeros(self.n_features, dtype=np.int64)

    def update(self, codes_flat: torch.Tensor) -> None:
        """Update from a flat ``(N, F)`` codes tensor (one chunk's samples)."""
        # GPU reductions, then move scalars to CPU.
        chunk_sum = codes_flat.sum(dim=0).cpu().numpy().astype(np.float64)
        chunk_sumsq = (codes_flat * codes_flat).sum(dim=0).cpu().numpy().astype(np.float64)
        chunk_max = codes_flat.max(dim=0).values.cpu().numpy().astype(np.float32)
        chunk_active = (codes_flat > 0).sum(dim=0).cpu().numpy().astype(np.int64)
        self.sum += chunk_sum
        self.sumsq += chunk_sumsq
        np.maximum(self.max, chunk_max, out=self.max)
        self.active_count += chunk_active
        self.total_count += codes_flat.shape[0]

    def finalize(self) -> dict[str, np.ndarray]:
        n = max(self.total_count, 1)
        mean = self.sum / n
        var = np.maximum(self.sumsq / n - mean * mean, 0.0)
        active_n = np.maximum(self.active_count, 1)
        return {
            "mean": mean.astype(np.float32),
            "std": np.sqrt(var).astype(np.float32),
            "max": self.max,
            "frac_active": (self.active_count / n).astype(np.float32),
            "mean_when_active": (self.sum / active_n).astype(np.float32),
            "active_count": self.active_count,
            "total_count": np.int64(self.total_count),
        }


# ── Rich stats: tracked features ────────────────────────────────────────────


@dataclass
class _CondStats:
    """Running scalar stats for one feature × one condition (cell_line/pert)."""
    sum: float = 0.0
    sumsq: float = 0.0
    max: float = 0.0
    active_count: int = 0
    total_count: int = 0

    def update(self, vals: np.ndarray) -> None:
        self.sum += float(vals.sum())
        self.sumsq += float((vals * vals).sum())
        if vals.size:
            m = float(vals.max())
            if m > self.max:
                self.max = m
        self.active_count += int((vals > 0).sum())
        self.total_count += int(vals.size)

    def finalize(self) -> dict[str, float]:
        n = max(self.total_count, 1)
        mean = self.sum / n
        var = max(self.sumsq / n - mean * mean, 0.0)
        return {
            "mean": float(mean),
            "std": float(np.sqrt(var)),
            "max": self.max,
            "frac_active": self.active_count / n,
            "active_count": int(self.active_count),
            "total_count": int(self.total_count),
        }


@dataclass
class RichStats:
    """Per-tracked-feature accumulators for the deep card.

    ``feature_id`` is the index into the dictionary's code space.
    Gene aggregation uses ``(V,)`` arrays; per-cell-line and per-pert
    are dicts keyed by string label.
    """
    feature_id: int
    n_genes: int
    n_top_cells: int = 100

    # Per-gene: (V,) arrays
    gene_sum: np.ndarray = field(init=False)            # Σ f over (cell, token) at this gene
    gene_active_count: np.ndarray = field(init=False)   # count of tokens where this gene fired
    gene_total_count: np.ndarray = field(init=False)    # count of tokens at this gene (active or not)
    gene_max: np.ndarray = field(init=False)

    # Per-cell-line / per-pert marginals — string-keyed
    by_cell_line: dict[str, _CondStats] = field(default_factory=lambda: defaultdict(_CondStats))
    by_pert: dict[str, _CondStats] = field(default_factory=lambda: defaultdict(_CondStats))

    # Top-N activating cells: min-heap of (mean_f, cell_id, pert, cell_line)
    _cell_heap: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.gene_sum = np.zeros(self.n_genes, dtype=np.float64)
        self.gene_active_count = np.zeros(self.n_genes, dtype=np.int64)
        self.gene_total_count = np.zeros(self.n_genes, dtype=np.int64)
        self.gene_max = np.zeros(self.n_genes, dtype=np.float32)

    def _push_top_cell(self, mean_f: float, cell_id: str, pert: str, cell_line: str) -> None:
        entry = (mean_f, cell_id, pert, cell_line)
        if len(self._cell_heap) < self.n_top_cells:
            heapq.heappush(self._cell_heap, entry)
        elif mean_f > self._cell_heap[0][0]:
            heapq.heapreplace(self._cell_heap, entry)

    def update(
        self,
        f_per_token: np.ndarray,            # (n, T) for this feature on chunk
        gene_dataset_ids: np.ndarray,        # (n, T)
        cell_id: np.ndarray,                 # (n,)
        pert: np.ndarray,                    # (n,)
        cell_line: np.ndarray,               # (n,)
    ) -> None:
        n, T = f_per_token.shape
        if gene_dataset_ids.shape != (n, T):
            raise ValueError(
                f"gene_dataset_ids shape {gene_dataset_ids.shape} != ({n}, {T})"
            )

        # Per-gene scatter (vectorized via bincount).
        g_flat = gene_dataset_ids.ravel()
        f_flat = f_per_token.ravel()
        is_active = f_flat > 0
        self.gene_sum += np.bincount(g_flat, weights=f_flat, minlength=self.n_genes)
        self.gene_active_count += np.bincount(
            g_flat, weights=is_active.astype(np.int64), minlength=self.n_genes
        ).astype(np.int64)
        self.gene_total_count += np.bincount(g_flat, minlength=self.n_genes)
        # Per-gene max — done via per-cell loop because numpy has no scatter-max.
        # f_flat values for tokens with the same gene are merged via maximum.
        # Use np.maximum.at, which is slow but only runs over (cell, token).
        np.maximum.at(self.gene_max, g_flat, f_flat.astype(np.float32))

        # Per-cell-line and per-pert marginals — group by string label.
        # For each unique label in this chunk, slice f_per_token rows and
        # update the corresponding _CondStats.
        for cl_label in np.unique(cell_line):
            mask = cell_line == cl_label
            self.by_cell_line[str(cl_label)].update(f_per_token[mask].ravel())
        for p_label in np.unique(pert):
            mask = pert == p_label
            self.by_pert[str(p_label)].update(f_per_token[mask].ravel())

        # Top-N cells by mean activation across tokens.
        cell_means = f_per_token.mean(axis=1)
        for c in range(n):
            self._push_top_cell(
                float(cell_means[c]), str(cell_id[c]), str(pert[c]), str(cell_line[c])
            )

    def top_genes(self, gene_symbols: np.ndarray, k: int = 50) -> list[dict]:
        """Top-k genes by mean activation when present (sum / total_count)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_when_present = np.where(
                self.gene_total_count > 0,
                self.gene_sum / np.maximum(self.gene_total_count, 1),
                0.0,
            )
        order = np.argsort(-mean_when_present)[:k]
        return [
            {
                "gene_id": int(g),
                "gene_symbol": str(gene_symbols[g]),
                "mean_when_present": float(mean_when_present[g]),
                "mean_when_active": (
                    float(self.gene_sum[g] / self.gene_active_count[g])
                    if self.gene_active_count[g] > 0
                    else 0.0
                ),
                "max": float(self.gene_max[g]),
                "frac_active": (
                    float(self.gene_active_count[g] / self.gene_total_count[g])
                    if self.gene_total_count[g] > 0
                    else 0.0
                ),
                "n_cells_active": int(self.gene_active_count[g]),
                "n_cells_total": int(self.gene_total_count[g]),
            }
            for g in order
        ]

    def top_cells(self) -> list[dict]:
        """Top-N activating cells, sorted descending by mean f."""
        return [
            {"cell_id": cid, "pert": p, "cell_line": cl, "mean_f": v}
            for v, cid, p, cl in sorted(self._cell_heap, reverse=True)
        ]

    def finalize(self, gene_symbols: np.ndarray, *, top_k_genes: int = 50) -> dict:
        return {
            "feature_id": self.feature_id,
            "top_cells": self.top_cells(),
            "top_genes": self.top_genes(gene_symbols, k=top_k_genes),
            "by_cell_line": {k: v.finalize() for k, v in self.by_cell_line.items()},
            "by_pert": {k: v.finalize() for k, v in self.by_pert.items()},
        }


# ── Aggregator: drives both levels in one pass ──────────────────────────────


@dataclass
class Aggregator:
    """Two-level aggregator over chunks. Single pass.

    Args:
        n_features: dictionary code dimension (e.g. 4096).
        tracked_features: feature ids to build rich stats for (e.g. top-50
            ESM-specific). Only these get per-gene + per-condition stats.
        n_genes: gene vocab size (from `meta/gene_symbols`).
    """
    n_features: int
    tracked_features: list[int]
    n_genes: int
    scalar: ScalarStats = field(init=False)
    rich: dict[int, RichStats] = field(init=False)

    def __post_init__(self) -> None:
        self.scalar = ScalarStats(n_features=self.n_features)
        self.rich = {
            f: RichStats(feature_id=int(f), n_genes=self.n_genes)
            for f in self.tracked_features
        }

    def update(self, chunk: Chunk) -> None:
        """Update both accumulator levels from one chunk."""
        n, T, F = chunk.codes.shape
        codes_flat = chunk.codes.reshape(n * T, F)

        # Scalar pass: vectorized over all features on GPU.
        self.scalar.update(codes_flat)

        # Rich pass: per tracked feature, slice f and aggregate on CPU.
        codes_cpu = chunk.codes.detach().to("cpu").to(torch.float32)
        for feat in self.tracked_features:
            f_per_token = codes_cpu[..., feat].numpy()    # (n, T)
            self.rich[feat].update(
                f_per_token=f_per_token,
                gene_dataset_ids=chunk.gene_dataset_ids,
                cell_id=chunk.cell_id,
                pert=chunk.pert,
                cell_line=chunk.cell_line,
            )

    def finalize_index(self) -> dict[str, np.ndarray]:
        """Cheap-index dict of per-feature scalar arrays (length n_features)."""
        return self.scalar.finalize()

    def finalize_cards(self, gene_symbols: np.ndarray, *, top_k_genes: int = 50) -> dict[int, dict]:
        """Per-tracked-feature rich finalized dict, keyed by feature_id."""
        return {f: self.rich[f].finalize(gene_symbols, top_k_genes=top_k_genes) for f in self.tracked_features}
