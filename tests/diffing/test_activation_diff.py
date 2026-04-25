from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.diffing.cli import main as cli_main
from scripts.diffing.pairs import load_pair
from scripts.diffing.base import get
from scripts.interp.hook_readers import H5ActivationReader
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


def _write_activations(
    path: Path, capture: str, tensor: torch.Tensor, labels: dict | None = None
) -> None:
    with H5ActivationSink(
        path,
        runner="test_runner",
        dataset="test",
        split="train",
        capture_names=[capture],
    ) as sink:
        sink.write(
            ActivationRecord(
                name=capture,
                tensor=tensor,
                metadata_tags={},
                per_cell=labels or {},
                layout="BD",
            )
        )


def test_activation_diff_known_answer(tmp_path: Path) -> None:
    capture = "layer_0"
    torch.manual_seed(0)
    n, d = 16, 8
    a = torch.randn(n, d)
    b = a + torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * n)

    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    path_out = tmp_path / "scores.h5"
    _write_activations(path_a, capture, a, labels={"cell_id": np.arange(n)})
    _write_activations(path_b, capture, b)

    cli_main(
        [
            "--method", "activation_diff",
            "--a", str(path_a),
            "--b", str(path_b),
            "--capture", capture,
            "--out", str(path_out),
            "--relationship", "finetune_vs_base",
        ]
    )

    with H5ActivationReader(path_out) as r:
        l2, l2_labels = r.read("l2_per_cell")
        cosine, _ = r.read("cosine_per_cell")
        norm_a, _ = r.read("norm_a_per_cell")
        norm_b, _ = r.read("norm_b_per_cell")
        relative, _ = r.read("relative_diff")
        mean_abs, _ = r.read("mean_abs_per_dim")

    assert l2.shape == (n,)
    assert cosine.shape == (n,)
    assert norm_a.shape == (n,)
    assert norm_b.shape == (n,)
    assert relative.shape == (n,)
    assert mean_abs.shape == (1, d)

    assert torch.allclose(l2, torch.full((n,), 1.0), atol=1e-5)
    assert torch.allclose(norm_a, a.norm(dim=-1), atol=1e-5)
    assert torch.allclose(norm_b, b.norm(dim=-1), atol=1e-5)
    expected_relative = l2 / (norm_a + norm_b)
    assert torch.allclose(relative, expected_relative, atol=1e-5)

    expected_mean_abs = torch.zeros(d)
    expected_mean_abs[0] = 1.0
    assert torch.allclose(mean_abs.squeeze(0), expected_mean_abs, atol=1e-5)

    assert "cell_id" in l2_labels
    assert np.array_equal(np.asarray(l2_labels["cell_id"]), np.arange(n))


def test_activation_diff_streams_across_chunks(tmp_path: Path) -> None:
    capture = "layer_0"
    n, d = 10_000, 4
    a = torch.zeros(n, d)
    b = torch.ones(n, d)

    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    path_out = tmp_path / "scores.h5"
    _write_activations(path_a, capture, a)
    _write_activations(path_b, capture, b)

    cli_main(
        [
            "--method", "activation_diff",
            "--a", str(path_a),
            "--b", str(path_b),
            "--capture", capture,
            "--out", str(path_out),
        ]
    )

    with H5ActivationReader(path_out) as r:
        l2, _ = r.read("l2_per_cell")
        mean_abs, _ = r.read("mean_abs_per_dim")

    assert l2.shape == (n,)
    assert torch.allclose(l2, torch.full((n,), float(d) ** 0.5), atol=1e-5)
    assert torch.allclose(mean_abs.squeeze(0), torch.ones(d), atol=1e-5)


def test_pair_rejects_row_mismatch(tmp_path: Path) -> None:
    capture = "layer_0"
    _write_activations(tmp_path / "a.h5", capture, torch.zeros(10, 4))
    _write_activations(tmp_path / "b.h5", capture, torch.zeros(12, 4))

    with pytest.raises(ValueError, match="row mismatch"):
        load_pair(tmp_path / "a.h5", tmp_path / "b.h5", capture=capture)


def test_registry_lists_activation_diff() -> None:
    import scripts.diffing.methods  # noqa: F401 — ensures registration
    from scripts.diffing.base import registered_methods

    assert "activation_diff" in registered_methods()
    assert get("activation_diff").name == "activation_diff"
