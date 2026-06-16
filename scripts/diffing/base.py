"""Core types and registry for the diffing package.

A `DiffPair` names two sources of cached activations and the alignment
that brings them into a comparable space. A `DiffMethod` consumes a pair
and writes per-cell scores to an `H5ActivationSink` so downstream code
(visualize/, future methods) can read scores with the same reader it
uses for raw activations.

Sources are lazy: they hold h5 paths, not loaded tensors. Methods choose
between `iter_chunks` (streaming, default) and `load_all` (whole tensor
in RAM) and declare the choice via `streaming_ok`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterator, Literal

import h5py
import numpy as np
import torch

from scripts.interp.hook_sinks import H5ActivationSink, group_path


Relationship = Literal[
    "finetune_vs_base",
    "cross_model",
    "checkpoint_steps",
    "unspecified",
]


@dataclass(frozen=True)
class ActivationSource:
    """Lazy handle to cached activations. No tensors loaded until read.

    `max_rows` caps iteration at the first N rows in shard order. Set by
    `load_pair` when truncating the longer side to match the shorter,
    after verifying that the prefix `cell_id`s agree element-wise. Tests
    and direct-construction callers leave it None.
    """

    path: Path
    capture: str
    tags: dict[str, str] = field(default_factory=dict)
    max_rows: int | None = None

    def iter_chunks(
        self, chunk_rows: int
    ) -> Iterator[tuple[torch.Tensor, dict[str, np.ndarray]]]:
        """Yield `(activation_chunk, labels_chunk)` across shards.

        Chunks never span shard boundaries — a single shard read is the
        upper bound on concatenation cost per yield. Labels are returned
        as numpy (even non-string) to dodge torch's missing string dtype;
        callers convert if they need tensors. When `max_rows` is set,
        iteration stops once that many rows have been yielded.
        """
        act_path = f"{group_path(self.capture, self.tags)}/activation"
        labels_path = f"{group_path(self.capture, self.tags)}/labels"
        emitted = 0
        for shard_file in _shard_files(self.path):
            with h5py.File(shard_file, "r") as f:
                if act_path not in f:
                    continue
                dset = f[act_path]
                n = dset.shape[0]
                if self.max_rows is not None:
                    remaining = self.max_rows - emitted
                    if remaining <= 0:
                        return
                    n = min(n, remaining)
                label_names = (
                    list(f[labels_path].keys()) if labels_path in f else []
                )
                for start in range(0, n, chunk_rows):
                    stop = min(start + chunk_rows, n)
                    act = torch.from_numpy(dset[start:stop])
                    labels: dict[str, np.ndarray] = {}
                    for lname in label_names:
                        arr = f[f"{labels_path}/{lname}"][start:stop]
                        if arr.dtype == object:
                            arr = np.array(
                                [
                                    x.decode() if isinstance(x, bytes) else x
                                    for x in arr
                                ],
                                dtype=object,
                            )
                        labels[lname] = arr
                    yield act, labels
                    emitted += stop - start

    def load_all(self) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
        """Concatenate all chunks into a single in-memory tensor. Use only
        when the method genuinely needs the whole dataset — this is the
        RAM-unsafe path and callers must know their data fits."""
        acts: list[torch.Tensor] = []
        labels_parts: dict[str, list[np.ndarray]] = {}
        for act, labels in self.iter_chunks(chunk_rows=_DEFAULT_LOAD_CHUNK):
            acts.append(act)
            for k, v in labels.items():
                labels_parts.setdefault(k, []).append(v)
        if not acts:
            raise KeyError(
                f"no activation for capture={self.capture!r} tags={self.tags!r} "
                f"at {self.path}"
            )
        full = torch.cat(acts, dim=0)
        labels = {k: np.concatenate(v, axis=0) for k, v in labels_parts.items()}
        return full, labels

    def row_count(self) -> int:
        total = 0
        act_path = f"{group_path(self.capture, self.tags)}/activation"
        for shard_file in _shard_files(self.path):
            with h5py.File(shard_file, "r") as f:
                if act_path in f:
                    total += f[act_path].shape[0]
        if self.max_rows is not None:
            return min(total, self.max_rows)
        return total

    def feature_shape(self) -> tuple[int, ...]:
        act_path = f"{group_path(self.capture, self.tags)}/activation"
        for shard_file in _shard_files(self.path):
            with h5py.File(shard_file, "r") as f:
                if act_path in f:
                    return tuple(f[act_path].shape[1:])
        raise KeyError(f"no activation for capture={self.capture!r} at {self.path}")


_DEFAULT_LOAD_CHUNK = 4096


def _shard_files(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(path.glob(H5ActivationSink._SHARD_GLOB))
    return [path]


@dataclass(frozen=True)
class DiffPair:
    a: ActivationSource
    b: ActivationSource
    alignment: "Alignment"
    relationship: Relationship = "unspecified"

    def __post_init__(self) -> None:
        if self.a.row_count() != self.b.row_count():
            raise ValueError(
                f"DiffPair row mismatch: a has {self.a.row_count()} rows, "
                f"b has {self.b.row_count()}. Pairs must be row-aligned "
                "(same cells, same order) for any alignment to be meaningful."
            )


class DiffMethod(ABC):
    """Base class for diffing methods.

    Concrete methods set `name`, optionally flip `supports_cross_arch` /
    `streaming_ok` / `requires_preprocessing` / `requires_models`, and
    implement `score`. `fit` / `save` / `load` / `config_tag` all default
    to no-ops so stateless methods don't need to override them.
    """

    name: ClassVar[str]
    supports_cross_arch: ClassVar[bool] = False
    streaming_ok: ClassVar[bool] = True
    # True if score() reads pre-cached h5; False if it runs models live.
    requires_preprocessing: ClassVar[bool] = True
    # True if score() needs live model handles (logit-lens, causal patching).
    requires_models: ClassVar[bool] = False
    output_capture_names: ClassVar[list[str]] = []

    @classmethod
    def from_config(cls, cfg) -> "DiffMethod":
        """Build an instance from a config dict. Override in stateful methods
        to read sub-blocks (e.g. cfg.training.*). Default returns cls()."""
        return cls()

    def fit(self, pair: DiffPair) -> None:
        return None

    @abstractmethod
    def score(self, pair: DiffPair, sink: H5ActivationSink) -> None: ...

    def save(self, out_dir: Path) -> None:
        return None

    @classmethod
    def load(cls, out_dir: Path) -> "DiffMethod":
        return cls()

    def config_tag(self) -> str:
        """Leaf dir under <pair>/<method>/<capture>/ that namespaces reruns
        with different hyperparams (e.g. `n50_b_minus_a`). Override in
        stateful methods; default is fine for stateless ones."""
        return "default"


_REGISTRY: dict[str, type[DiffMethod]] = {}


def register(name: str):
    def decorator(cls: type[DiffMethod]) -> type[DiffMethod]:
        if name in _REGISTRY:
            raise ValueError(
                f"diffing method {name!r} already registered to "
                f"{_REGISTRY[name].__name__}"
            )
        cls.name = name
        _REGISTRY[name] = cls
        return cls

    return decorator


def get(name: str) -> type[DiffMethod]:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown diffing method {name!r}. Registered: "
            f"{sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


def registered_methods() -> list[str]:
    return sorted(_REGISTRY.keys())


from scripts.diffing.alignment import Alignment  # noqa: E402  (resolve forward ref)
