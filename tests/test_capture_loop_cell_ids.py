from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from nnsight import NNsight

from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import HookManager


class _Tiny(nn.Module):
    def __init__(self, d: int = 4) -> None:
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def test_cell_ids_unique_across_batches(tmp_path: Path) -> None:
    # HookManager + H5ActivationSink must aggregate per-cell sidecar columns
    # across multiple hm.run() calls into one contiguous axis-0 column —
    # e.g. cell_ids set per batch via set_per_cell end up concatenated in
    # the h5 dataset in call order, not overwriting each other.
    model = _Tiny().eval()
    nn_model = NNsight(model)
    path = tmp_path / "act.h5"
    batch_sizes = [3, 5, 2]
    total = sum(batch_sizes)

    with H5ActivationSink(
        path, runner="t", dataset="t", split="t", capture_names=["lin"]
    ) as sink, HookManager(
        nn_model, capture=[("lin", lambda m: m.lin.output)], sink=sink
    ) as hm:
        cell_offset = 0
        for bs in batch_sizes:
            hm.set_per_cell({
                "cell_id": torch.arange(cell_offset, cell_offset + bs),
            })
            hm.run(torch.randn(bs, 4))
            cell_offset += bs

    with h5py.File(path, "r") as f:
        cell_ids = f["lin/labels/cell_id"][...]
    assert cell_ids.shape == (total,)
    assert np.array_equal(cell_ids, np.arange(total))
    assert len(set(cell_ids)) == total