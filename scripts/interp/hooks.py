from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class ActivationRecord:
    name: str
    tensor: torch.Tensor
    metadata_tags: dict[str, str]
    # Per-cell sidecars aligned on axis 0 of `tensor`. Typical uses: cell_id,
    # gene_id, perturbation label, split index. Downstream probing / SAE code
    # reads these alongside the activation to recover which row came from
    # where without having to re-join by order.
    per_cell: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.tensor.shape[0]
        for k, v in self.per_cell.items():
            if v.shape[0] != n:
                raise ValueError(
                    f"per_cell[{k!r}] has axis-0 length {v.shape[0]}, "
                    f"expected {n} to match tensor.shape[0]"
                )


# Sinks are duck-typed: any object with `write(record) / close() / context-
# manager protocol` works. MemoryActivationSink and H5ActivationSink are the
# two in-repo implementations.
CaptureSpec = tuple[str, Callable[[Any], Any]]


class HookManager:
    def __init__(
        self,
        nn_model: Any,
        capture: list[CaptureSpec],
        sink: Any,
        capture_dtype: torch.dtype = torch.float32,
        gate: Callable[[dict[str, str]], bool] | None = None,
    ) -> None:
        self.nn_model = nn_model
        self.capture = capture
        self.sink = sink
        self.capture_dtype = capture_dtype
        self.gate = gate
        self.current_tags: dict[str, str] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.current_tags[key] = value

    def __enter__(self) -> HookManager:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        try:
            self.sink.close()
        except Exception:
            # A close failure during a propagating exception would mask the
            # original. Swallow so the caller sees the real error.
            if exc_type is None:
                raise

    # The `with self.nn_model.trace(...):` block below must be syntactic:
    # nnsight 0.5's InterleavingTracer parses the caller's source AST at
    # enter-time and raises WithBlockNotFoundError if the block is split
    # across manual __enter__/__exit__ calls or yielded from a generator.
    # `torch.no_grad` wraps the trace because pure activation capture does
    # not need autograd graph — skipping it avoids useless memory retention
    # and makes captures match no_grad replays bit-for-bit.
    def run(self, *args: Any, **kwargs: Any) -> None:
        saved: list[tuple[str, Any]] = []
        with torch.no_grad(), self.nn_model.trace(*args, **kwargs):
            for name, accessor in self.capture:
                saved.append((name, accessor(self.nn_model).save()))
        tags = dict(self.current_tags)
        if self.gate is not None and not self.gate(tags):
            return
        for name, s in saved:
            value = getattr(s, "value", s)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"Capture target {name!r} produced {type(value).__name__}, "
                    "expected torch.Tensor — accessor must resolve to a tensor "
                    "(e.g. `.output[0]` for tuple-returning modules)."
                )
            # `copy=True` guarantees the stored tensor is independent of the
            # source proxy/buffer even on fp32→fp32 CPU no-ops, so downstream
            # sinks cannot be invalidated by subsequent forwards reusing
            # memory.
            tensor = value.detach().to(
                "cpu", dtype=self.capture_dtype, copy=True
            )
            self.sink.write(
                ActivationRecord(name=name, tensor=tensor, metadata_tags=tags)
            )
