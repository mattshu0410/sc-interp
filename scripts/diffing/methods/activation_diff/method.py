"""Stateless activation-level diffing.

Per-cell outputs (cell-aligned with pair.a):
    l2_per_cell        ||b - a||_2
    cosine_per_cell    cos(a, b)
    norm_a_per_cell    ||a||_2
    norm_b_per_cell    ||b||_2
    relative_diff      ||b - a||_2 / (||a||_2 + ||b||_2)

Per-feature summary, written once at end:
    mean_abs_per_dim   mean_i |b_i - a_i| along axis 0

Streams through chunks so memory is O(chunk_rows).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from scripts.diffing.base import DiffMethod, DiffPair, register
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


_CHUNK_ROWS = 4096


@register("activation_diff")
class ActivationDiff(DiffMethod):
    supports_cross_arch = False
    streaming_ok = True
    output_capture_names = [
        "l2_per_cell",
        "cosine_per_cell",
        "norm_a_per_cell",
        "norm_b_per_cell",
        "relative_diff",
        "mean_abs_per_dim",
    ]

    def score(self, pair: DiffPair, sink: H5ActivationSink) -> None:
        a_iter = pair.a.iter_chunks(_CHUNK_ROWS)
        b_iter = pair.b.iter_chunks(_CHUNK_ROWS)

        total_abs_delta: torch.Tensor | None = None
        total_rows = 0

        for (a_chunk, a_labels), (b_chunk, _) in zip(a_iter, b_iter):
            a_aligned, b_aligned = pair.alignment.apply(a_chunk, b_chunk)
            if a_aligned.shape != b_aligned.shape:
                raise ValueError(
                    f"activation_diff requires pointwise-shape-matching chunks "
                    f"after alignment; got a={tuple(a_aligned.shape)} "
                    f"b={tuple(b_aligned.shape)}. This method does not support "
                    "cross-arch pairs."
                )

            delta = b_aligned - a_aligned
            # Flatten feature dims so (N, T, d) layouts reduce over T+d
            # together — "how different is the entire per-cell forward."
            flat_delta = delta.reshape(delta.shape[0], -1)
            flat_a = a_aligned.reshape(a_aligned.shape[0], -1)
            flat_b = b_aligned.reshape(b_aligned.shape[0], -1)

            l2 = flat_delta.norm(dim=-1)
            cosine = F.cosine_similarity(flat_a, flat_b, dim=-1)
            norm_a = flat_a.norm(dim=-1)
            norm_b = flat_b.norm(dim=-1)
            denom = norm_a + norm_b
            relative = torch.where(denom > 0, l2 / denom, torch.zeros_like(l2))

            # Labels from A ride on l2_per_cell so downstream joins don't
            # need a second h5 pass.
            def _rec(name, tensor, per_cell=None):
                return ActivationRecord(
                    name=name, tensor=tensor, metadata_tags={},
                    per_cell=per_cell or {}, layout="BD",
                )

            sink.write(_rec("l2_per_cell", l2, per_cell=a_labels))
            sink.write(_rec("cosine_per_cell", cosine))
            sink.write(_rec("norm_a_per_cell", norm_a))
            sink.write(_rec("norm_b_per_cell", norm_b))
            sink.write(_rec("relative_diff", relative))
            sink.batch_end()

            abs_delta_sum = delta.abs().reshape(delta.shape[0], -1).sum(dim=0)
            total_abs_delta = (
                abs_delta_sum
                if total_abs_delta is None
                else total_abs_delta + abs_delta_sum
            )
            total_rows += delta.shape[0]

        if total_abs_delta is None or total_rows == 0:
            raise ValueError(
                "activation_diff produced no chunks — are a and b empty or "
                "do the captures not exist in both h5 files?"
            )

        mean_abs = (total_abs_delta / total_rows).unsqueeze(0)
        sink.write(
            ActivationRecord(
                name="mean_abs_per_dim",
                tensor=mean_abs,
                metadata_tags={},
                per_cell={},
                layout="BD",
            )
        )
        sink.batch_end()


# Intentionally not in iteration 1: sign-preserving per-cell-type mean-delta
# collectors; top-K max-activating-cell tracking. Both land when we have
# per-cell-type labels wired through the visualize/ pipeline.
