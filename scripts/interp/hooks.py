from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union

import numpy as np
import torch


# Closed vocabulary for activation tensor axis layouts, written as the h5
# dataset attr `layout` so probes/SAE code can branch on shape semantics
# without guessing from ndim. Extend when a new layout is genuinely needed —
# don't encode layouts as ad-hoc strings at callsites.
Layout = Literal["", "BTD", "BD"]


@dataclass(frozen=True)
class ActivationRecord:
    name: str
    tensor: torch.Tensor
    metadata_tags: dict[str, str]
    # Per-cell sidecars aligned on axis 0 of `tensor`. Typical uses: cell_id,
    # gene_id, perturbation label, split index. Downstream probing / SAE code
    # reads these alongside the activation to recover which row came from
    # where without having to re-join by order. np.ndarray is allowed so
    # string labels (dtype=object) can ride on the same channel — torch has
    # no string dtype.
    per_cell: dict[str, torch.Tensor | np.ndarray] = field(default_factory=dict)
    # "" means unspecified — valid when the caller doesn't care.
    layout: Layout = ""

    def __post_init__(self) -> None:
        n = self.tensor.shape[0]
        for k, v in self.per_cell.items():
            if v.shape[0] != n:
                raise ValueError(
                    f"per_cell[{k!r}] has axis-0 length {v.shape[0]}, "
                    f"expected {n} to match tensor.shape[0]"
                )


# Capture specs may optionally carry a Layout annotation as a 3rd slot:
# `("name", accessor)` or `("name", accessor, "BTD")`. Two shapes kept so
# existing 2-tuple callsites don't churn.
# Sinks are duck-typed: any object with `write(record) / close() / context-
# manager protocol` works. MemoryActivationSink and H5ActivationSink are the
# two in-repo implementations.
CaptureSpec = Union[
    tuple[str, Callable[[Any], Any]],
    tuple[str, Callable[[Any], Any], Layout],
]


class HookManager:
    def __init__(
        self,
        nn_model: Any,
        capture: list[CaptureSpec],
        sink: Any,
        capture_dtype: torch.dtype = torch.float32,
        gate: Callable[[dict[str, str]], bool] | None = None,
        no_grad: bool = True,
    ) -> None:
        self.nn_model = nn_model
        self.capture = capture
        self.sink = sink
        self.capture_dtype = capture_dtype
        self.gate = gate
        self.no_grad = no_grad
        self.current_tags: dict[str, str] = {}
        # Per-cell labels are cleared after every run(): batches carry
        # different cell ids, so silent reuse of stale ids would misalign
        # the h5 sidecar columns.
        self.current_per_cell: dict[str, torch.Tensor | np.ndarray] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.current_tags[key] = value

    def set_per_cell(
        self, per_cell: dict[str, torch.Tensor | np.ndarray]
    ) -> None:
        self.current_per_cell = dict(per_cell)

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

    # nnsight 0.5's InterleavingTracer parses the caller's source AST at
    # enter-time and raises WithBlockNotFoundError if the `with trace(...):`
    # block is split across manual __enter__/__exit__ calls or yielded from
    # a generator. Keep the block syntactic inside run().
    def run(self, *args: Any, **kwargs: Any) -> Any:
        saved: list[tuple[str, Any, Layout]] = []
        no_grad_ctx = torch.no_grad() if self.no_grad else contextlib.nullcontext()
        with no_grad_ctx, self.nn_model.trace(*args, **kwargs):
            for spec in self.capture:
                name, accessor = spec[0], spec[1]
                layout: Layout = spec[2] if len(spec) == 3 else ""
                saved.append((name, accessor(self.nn_model).save(), layout))
            # Save nn_model.output LAST — its provider fires last in the
            # forward, so this ordering is always valid. User captures are
            # saved above in their listed order (which must be forward
            # order, else OutOfOrderError). Kept as a separate local because
            # the model output can be a dict/tuple and must not flow through
            # the sink (ActivationRecord.tensor requires a Tensor).
            output_saved = self.nn_model.output.save()
        # nnsight 0.5 pushes saved values verbatim back into this frame —
        # `output_saved` is already the raw Tensor/dict/tuple, not a proxy.
        tags = dict(self.current_tags)
        per_cell = self.current_per_cell
        self.current_per_cell = {}
        # `gate` filters sink writes only; the model output is always returned
        # so predict-loop callers get the forward result regardless.
        if self.gate is not None and not self.gate(tags):
            return output_saved
        for name, s, layout in saved:
            if not isinstance(s, torch.Tensor):
                raise TypeError(
                    f"Capture target {name!r} produced {type(s).__name__}, "
                    "expected torch.Tensor — accessor must resolve to a tensor "
                    "(e.g. `.output[0]` for tuple-returning modules)."
                )
            # `copy=True` guarantees the stored tensor is independent of the
            # source proxy/buffer even on fp32→fp32 CPU no-ops, so downstream
            # sinks cannot be invalidated by subsequent forwards reusing
            # memory.
            tensor = s.detach().to("cpu", dtype=self.capture_dtype, copy=True)
            self.sink.write(
                ActivationRecord(
                    name=name,
                    tensor=tensor,
                    metadata_tags=tags,
                    per_cell=per_cell,
                    layout=layout,
                )
            )
        return output_saved
