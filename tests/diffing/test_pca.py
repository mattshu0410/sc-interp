from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.diffing.base import get, registered_methods
from scripts.diffing.methods.pca.method import PCA
from scripts.diffing.pairs import load_pair
from scripts.interp.hook_readers import H5ActivationReader
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


CAPTURE = "layer_0"


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="no cuda"
            ),
        ),
    ]
)
def device(request) -> str:
    return request.param


def _write_activations(path: Path, tensor: torch.Tensor, labels: dict | None = None) -> None:
    with H5ActivationSink(
        path,
        runner="test",
        dataset="test",
        split="train",
        capture_names=[CAPTURE],
    ) as sink:
        sink.write(
            ActivationRecord(
                name=CAPTURE,
                tensor=tensor,
                metadata_tags={},
                per_cell=labels or {},
                layout="BD",
            )
        )


def _sink(path: Path) -> H5ActivationSink:
    return H5ActivationSink(
        path,
        runner="pca_test",
        dataset="pca_test",
        split="diff",
        capture_names=["pc_scores", "explained_variance", "explained_variance_ratio"],
    )


def test_pca_recovers_planted_direction(tmp_path: Path, device: str) -> None:
    torch.manual_seed(0)
    n, d = 512, 16
    a = torch.randn(n, d)
    alpha = torch.linspace(-1.0, 1.0, n).unsqueeze(1)
    planted = torch.zeros(d)
    planted[0] = 1.0
    b = a + alpha * planted

    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    path_out = tmp_path / "pca.h5"
    _write_activations(path_a, a, labels={"cell_id": np.arange(n)})
    _write_activations(path_b, b)

    pair = load_pair(path_a, path_b, capture=CAPTURE)
    method = PCA(n_components=4, batch_size=128, device=device)
    method.fit(pair)
    with _sink(path_out) as sink:
        method.score(pair, sink)

    pc1 = method._ipca.components_[0].cpu()
    assert abs(abs(float(pc1[0])) - 1.0) < 1e-3, f"PC1 not aligned with planted axis: {pc1}"
    assert float(method._ipca.explained_variance_ratio_[0]) > 0.99

    with H5ActivationReader(path_out) as r:
        pc_scores, labels = r.read("pc_scores")
        evr, _ = r.read("explained_variance_ratio")

    assert pc_scores.shape == (n, 4)
    assert evr.shape == (1, 4)
    assert np.array_equal(np.asarray(labels["cell_id"]), np.arange(n))


def test_pca_save_load_roundtrip(tmp_path: Path, device: str) -> None:
    torch.manual_seed(1)
    n, d = 300, 8
    a = torch.randn(n, d)
    b = a + torch.randn(n, d) * 0.5

    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    _write_activations(path_a, a)
    _write_activations(path_b, b)
    pair = load_pair(path_a, path_b, capture=CAPTURE)

    fitted = PCA(n_components=3, batch_size=100, device=device)
    fitted.fit(pair)
    fitted.save(tmp_path / "ckpt")

    loaded = PCA.load(tmp_path / "ckpt", device=device)
    assert loaded.n_components == 3
    assert loaded.target == "b_minus_a"
    np.testing.assert_allclose(
        loaded._ipca.components_.cpu(), fitted._ipca.components_.cpu(), atol=1e-6
    )
    np.testing.assert_allclose(
        loaded._ipca.mean_.cpu(), fitted._ipca.mean_.cpu(), atol=1e-6
    )
    np.testing.assert_allclose(
        loaded._ipca.explained_variance_ratio_.cpu(),
        fitted._ipca.explained_variance_ratio_.cpu(),
        atol=1e-6,
    )

    out_fit = tmp_path / "fit.h5"
    out_load = tmp_path / "load.h5"
    with _sink(out_fit) as sink:
        fitted.score(pair, sink)
    with _sink(out_load) as sink:
        loaded.score(pair, sink)

    with H5ActivationReader(out_fit) as r:
        fit_scores, _ = r.read("pc_scores")
    with H5ActivationReader(out_load) as r:
        load_scores, _ = r.read("pc_scores")
    torch.testing.assert_close(fit_scores, load_scores, atol=1e-5, rtol=1e-5)


def test_pca_target_selection(tmp_path: Path, device: str) -> None:
    torch.manual_seed(2)
    n, d = 200, 8
    mean_a = torch.full((d,), 5.0)
    mean_b = torch.full((d,), -3.0)
    a = mean_a + torch.randn(n, d) * 0.5
    b = mean_b + torch.randn(n, d) * 0.5

    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    _write_activations(path_a, a)
    _write_activations(path_b, b)
    pair = load_pair(path_a, path_b, capture=CAPTURE)

    fits = {
        "a": PCA(n_components=2, target="a", device=device),
        "b": PCA(n_components=2, target="b", device=device),
        "b_minus_a": PCA(n_components=2, target="b_minus_a", device=device),
        "a_minus_b": PCA(n_components=2, target="a_minus_b", device=device),
    }
    for m in fits.values():
        m.fit(pair)

    np.testing.assert_allclose(
        fits["a"]._ipca.mean_.cpu(), mean_a.numpy(), atol=0.1
    )
    np.testing.assert_allclose(
        fits["b"]._ipca.mean_.cpu(), mean_b.numpy(), atol=0.1
    )
    np.testing.assert_allclose(
        fits["b_minus_a"]._ipca.mean_.cpu(), (mean_b - mean_a).numpy(), atol=0.1
    )
    np.testing.assert_allclose(
        fits["a_minus_b"]._ipca.mean_.cpu(), (mean_a - mean_b).numpy(), atol=0.1
    )


def test_pca_registered() -> None:
    import scripts.diffing.methods  # noqa: F401

    assert "pca" in registered_methods()
    assert get("pca") is PCA
