"""PCA of activation differences.

fit() fits principal components on a chosen target tensor derived from
the pair, streaming through chunks. score() projects each chunk into
the fitted subspace and writes per-cell PC scores.

target selects what gets decomposed:
    b_minus_a   (default)   b - a
    a_minus_b                a - b
    a                        a
    b                        b

Per-cell outputs (cell-aligned with pair.a):
    pc_scores                 (N, n_components)   centered target @ components.T
Summary outputs, written once at end:
    explained_variance        (1, n_components)
    explained_variance_ratio  (1, n_components)

Streaming via IncrementalPCA.partial_fit + transform, so peak memory is
O(chunk_rows × features) regardless of total pair size.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from scripts.diffing.base import DiffMethod, DiffPair, register
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


_CHUNK_ROWS = 4096
_CHECKPOINT_FILENAME = "pca_model.pkl"

Target = Literal["b_minus_a", "a_minus_b", "a", "b"]


def _select_target(a: torch.Tensor, b: torch.Tensor, target: Target) -> torch.Tensor:
    if target == "b_minus_a":
        return b - a
    if target == "a_minus_b":
        return a - b
    if target == "a":
        return a
    if target == "b":
        return b
    raise ValueError(f"unknown PCA target {target!r}")


@register("pca")
class PCA(DiffMethod):
    supports_cross_arch = False
    streaming_ok = True
    output_capture_names = ["pc_scores", "explained_variance", "explained_variance_ratio"]

    def __init__(
        self,
        n_components: int = 50,
        batch_size: int = _CHUNK_ROWS,
        target: Target = "b_minus_a",
    ):
        self.n_components = n_components
        self.batch_size = batch_size
        self.target = target
        self._ipca = None  # sklearn.decomposition.IncrementalPCA

    @classmethod
    def from_config(cls, cfg) -> "PCA":
        t = cfg.training
        return cls(
            n_components=t.n_components,
            batch_size=t.batch_size,
            target=t.target,
        )

    def _new_ipca(self):
        from sklearn.decomposition import IncrementalPCA
        return IncrementalPCA(
            n_components=self.n_components, batch_size=self.batch_size
        )

    def _iter_target_chunks(self, pair: DiffPair):
        a_iter = pair.a.iter_chunks(self.batch_size)
        b_iter = pair.b.iter_chunks(self.batch_size)
        for (a_c, a_labels), (b_c, _) in zip(a_iter, b_iter):
            a_al, b_al = pair.alignment.apply(a_c, b_c)
            tgt = _select_target(a_al, b_al, self.target)
            flat = tgt.reshape(tgt.shape[0], -1).numpy()
            yield flat, a_labels

    def fit(self, pair: DiffPair) -> None:
        self._ipca = self._new_ipca()
        for batch, _ in self._iter_target_chunks(pair):
            # sklearn's IncrementalPCA requires the *first* partial_fit batch
            # to contain >= n_components rows; later batches can be smaller.
            # Skip under-sized chunks before the first fit; after that, any
            # size is accepted.
            if not hasattr(self._ipca, "components_") and batch.shape[0] < self.n_components:
                continue
            self._ipca.partial_fit(batch)
        if not hasattr(self._ipca, "components_"):
            raise RuntimeError(
                f"No chunk of size >= n_components={self.n_components} seen during "
                "fit. Decrease n_components or increase iter_chunks size."
            )

    def score(self, pair: DiffPair, sink: H5ActivationSink) -> None:
        if self._ipca is None or not hasattr(self._ipca, "components_"):
            raise RuntimeError("PCA.score() called before fit() or load()")

        for batch, a_labels in self._iter_target_chunks(pair):
            pc_scores = self._ipca.transform(batch)
            sink.write(
                ActivationRecord(
                    name="pc_scores",
                    tensor=torch.from_numpy(pc_scores).float(),
                    metadata_tags={},
                    per_cell=a_labels,
                    layout="BD",
                )
            )
            sink.batch_end()

        evr = torch.from_numpy(self._ipca.explained_variance_ratio_).float().unsqueeze(0)
        ev = torch.from_numpy(self._ipca.explained_variance_).float().unsqueeze(0)
        sink.write(
            ActivationRecord(
                name="explained_variance_ratio",
                tensor=evr,
                metadata_tags={},
                per_cell={},
                layout="BD",
            )
        )
        sink.write(
            ActivationRecord(
                name="explained_variance",
                tensor=ev,
                metadata_tags={},
                per_cell={},
                layout="BD",
            )
        )
        sink.batch_end()

    def save(self, out_dir: Path) -> None:
        if self._ipca is None or not hasattr(self._ipca, "components_"):
            raise RuntimeError("PCA.save() called before fit()")

        # Match upstream's pickled state dict schema so models can be moved
        # between projects with minimal translation.
        state = {
            "components_": self._ipca.components_,
            "explained_variance_": self._ipca.explained_variance_,
            "explained_variance_ratio_": self._ipca.explained_variance_ratio_,
            "mean_": self._ipca.mean_,
            "var_": getattr(self._ipca, "var_", None),
            "noise_variance_": getattr(self._ipca, "noise_variance_", None),
            "n_components": self._ipca.n_components,
            "n_samples_seen_": int(self._ipca.n_samples_seen_),
            "batch_size": self.batch_size,
            "target": self.target,
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / _CHECKPOINT_FILENAME, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, out_dir: Path) -> "PCA":
        with open(out_dir / _CHECKPOINT_FILENAME, "rb") as f:
            state = pickle.load(f)

        obj = cls(
            n_components=int(state["n_components"]),
            batch_size=int(state["batch_size"]),
            target=state["target"],
        )
        ipca = obj._new_ipca()
        ipca.components_ = _as_numpy(state["components_"])
        ipca.explained_variance_ = _as_numpy(state["explained_variance_"])
        ipca.explained_variance_ratio_ = _as_numpy(state["explained_variance_ratio_"])
        ipca.mean_ = _as_numpy(state["mean_"])
        if state.get("var_") is not None:
            ipca.var_ = _as_numpy(state["var_"])
        ipca.noise_variance_ = state.get("noise_variance_")
        ipca.n_components_ = int(state["n_components"])
        ipca.n_samples_seen_ = int(state["n_samples_seen_"])
        ipca.batch_size_ = int(state["batch_size"])
        obj._ipca = ipca
        return obj


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.asarray(x)
