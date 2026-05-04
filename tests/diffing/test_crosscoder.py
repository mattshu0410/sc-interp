from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("dictionary_learning")

from scripts.diffing.base import get, registered_methods  # noqa: E402
from scripts.diffing.methods.crosscoder import CrossCoderMethod  # noqa: E402
from scripts.diffing.pairs import load_pair  # noqa: E402
from scripts.interp.hook_readers import H5ActivationReader  # noqa: E402
from scripts.interp.hook_sinks import H5ActivationSink  # noqa: E402
from scripts.interp.hooks import ActivationRecord  # noqa: E402


CAPTURE = "layer_0"


def _write_activations(
    path: Path,
    tensor: torch.Tensor,
    *,
    labels: dict | None = None,
    layout: str = "BD",
) -> None:
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
                layout=layout,
            )
        )


def _sink(path: Path) -> H5ActivationSink:
    return H5ActivationSink(
        path,
        runner="cc_test",
        dataset="cc_test",
        split="diff",
        capture_names=CrossCoderMethod.output_capture_names,
    )


@pytest.fixture
def bd_pair(tmp_path: Path):
    torch.manual_seed(0)
    n, d = 256, 8
    a = torch.randn(n, d)
    delta = torch.zeros(n, d)
    delta[:, 0] = 1.0  # planted "B-specific" axis
    b = a + delta
    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    _write_activations(path_a, a, labels={"cell_id": np.arange(n)})
    _write_activations(path_b, b)
    return path_a, path_b, n, d


@pytest.fixture
def btd_pair(tmp_path: Path):
    torch.manual_seed(1)
    n, t, d = 64, 4, 8
    a = torch.randn(n, t, d)
    b = a + torch.randn(n, t, d) * 0.2
    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    _write_activations(path_a, a, labels={"cell_id": np.arange(n)}, layout="BTD")
    _write_activations(path_b, b, layout="BTD")
    return path_a, path_b, n, t, d


def test_crosscoder_registered() -> None:
    assert "crosscoder" in registered_methods()
    assert get("crosscoder") is CrossCoderMethod


def test_config_tag_variants() -> None:
    relu = CrossCoderMethod(model_type="relu", expansion_factor=4, l1_penalty=0.05, lr=2e-4, steps=500)
    topk = CrossCoderMethod(model_type="batch-top-k", expansion_factor=8, k=16, lr=1e-3, steps=1000)
    assert relu.config_tag() == "relu_x4_mu5e-02_lr2e-04_steps500"
    assert topk.config_tag() == "topk_x8_k16_lr1e-03_steps1000"


def test_unknown_model_type_rejected() -> None:
    with pytest.raises(ValueError, match="model_type"):
        CrossCoderMethod(model_type="bogus")


def test_normalizer_missing_raises_helpful(tmp_path: Path) -> None:
    # Capture written without running_stats — fit() should error pointing at backfill.
    n, d = 64, 4
    a = torch.randn(n, d)
    b = a + torch.randn(n, d) * 0.1
    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    with H5ActivationSink(
        path_a, runner="t", dataset="t", split="t",
        capture_names=[CAPTURE], compute_running_stats=False,
    ) as sink:
        sink.write(ActivationRecord(
            name=CAPTURE, tensor=a, metadata_tags={},
            per_cell={"cell_id": np.arange(n)}, layout="BD",
        ))
    with H5ActivationSink(
        path_b, runner="t", dataset="t", split="t",
        capture_names=[CAPTURE], compute_running_stats=False,
    ) as sink:
        sink.write(ActivationRecord(
            name=CAPTURE, tensor=b, metadata_tags={}, per_cell={}, layout="BD",
        ))
    pair = load_pair(path_a, path_b, capture=CAPTURE)
    method = CrossCoderMethod(steps=10, expansion_factor=2)
    with pytest.raises(FileNotFoundError, match="running_stats missing"):
        method.fit(pair)


def _read_tier1(path: Path) -> dict:
    with H5ActivationReader(path) as r:
        return {
            name: r.read(name)[0]
            for name in CrossCoderMethod.output_capture_names
        }


def test_fit_score_save_load_relu_bd(bd_pair, tmp_path: Path) -> None:
    path_a, path_b, n, d = bd_pair
    pair = load_pair(path_a, path_b, capture=CAPTURE)
    method = CrossCoderMethod(
        model_type="relu",
        expansion_factor=2,
        l1_penalty=0.001,
        steps=20,
        batch_size=64,
        chunk_rows=256,
        lr=1e-3,
        warmup_steps=5,
        seed=0,
        device="cpu",
        log_steps=1000,
    )
    method.fit(pair)
    assert method.model is not None
    assert method.activation_dim == d
    assert method.dict_size == 2 * d

    out_path = tmp_path / "scores.h5"
    with _sink(out_path) as sink:
        method.score(pair, sink)

    out = _read_tier1(out_path)
    for name in CrossCoderMethod.output_capture_names:
        assert out[name].shape == (1, 2 * d), name
    diff_a = out["feature_dec_norm_diff_a"].squeeze(0)
    diff_b = out["feature_dec_norm_diff_b"].squeeze(0)
    assert torch.all((diff_a >= 0) & (diff_a <= 1))
    assert torch.all((diff_b >= 0) & (diff_b <= 1))
    torch.testing.assert_close(diff_a + diff_b, torch.ones_like(diff_a), atol=1e-5, rtol=1e-5)

    ckpt_dir = tmp_path / "ckpt"
    method.save(ckpt_dir)
    assert (ckpt_dir / "model_final.pt").exists()
    assert (ckpt_dir / "config.json").exists()

    loaded = CrossCoderMethod.load(ckpt_dir)
    assert loaded.model_type == "relu"
    assert loaded.activation_dim == d
    assert loaded.dict_size == 2 * d

    out2 = tmp_path / "scores2.h5"
    with _sink(out2) as sink:
        loaded.score(pair, sink)
    out_loaded = _read_tier1(out2)
    for name in CrossCoderMethod.output_capture_names:
        torch.testing.assert_close(out_loaded[name], out[name], atol=1e-4, rtol=1e-4)


def test_fit_score_batch_topk_bd(bd_pair, tmp_path: Path) -> None:
    path_a, path_b, n, d = bd_pair
    pair = load_pair(path_a, path_b, capture=CAPTURE)
    method = CrossCoderMethod(
        model_type="batch-top-k",
        expansion_factor=2,
        k=4,
        steps=20,
        batch_size=64,
        chunk_rows=256,
        lr=1e-3,
        warmup_steps=5,
        threshold_start_step=1000,
        seed=0,
        device="cpu",
        log_steps=1000,
    )
    method.fit(pair)
    assert method.model is not None

    out_path = tmp_path / "scores.h5"
    with _sink(out_path) as sink:
        method.score(pair, sink)

    out = _read_tier1(out_path)
    for name in CrossCoderMethod.output_capture_names:
        assert out[name].shape == (1, 2 * d)


def test_fit_score_btd(btd_pair, tmp_path: Path) -> None:
    path_a, path_b, n, t, d = btd_pair
    pair = load_pair(path_a, path_b, capture=CAPTURE)
    method = CrossCoderMethod(
        model_type="relu",
        expansion_factor=2,
        l1_penalty=0.001,
        steps=10,
        batch_size=64,
        chunk_rows=64,
        lr=1e-3,
        warmup_steps=2,
        seed=0,
        device="cpu",
        log_steps=1000,
    )
    method.fit(pair)
    assert method.activation_dim == d
    assert method.dict_size == 2 * d

    out_path = tmp_path / "scores.h5"
    with _sink(out_path) as sink:
        method.score(pair, sink)
    out = _read_tier1(out_path)
    for name in CrossCoderMethod.output_capture_names:
        assert out[name].shape == (1, 2 * d), name


def test_from_config_round_trip() -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({
        "model": {
            "type": "batch-top-k",
            "code_normalization": "crosscoder",
            "same_init_for_all_layers": False,
            "norm_init_scale": 0.01,
            "init_with_transpose": True,
            "code_normalization_alpha_sae": 1.0,
            "code_normalization_alpha_cc": 0.1,
        },
        "training": {
            "expansion_factor": 4,
            "batch_size": 256,
            "chunk_rows": 64,
            "steps": 5000,
            "lr": 5.0e-4,
            "mu": 0.05,
            "k": 64,
            "auxk_alpha": 0.03125,
            "seed": 7,
        },
        "optimization": {
            "warmup_steps": 200,
            "decay_start": 4000,
            "threshold_beta": 0.999,
            "threshold_start_step": 500,
        },
    })
    m = CrossCoderMethod.from_config(cfg)
    assert m.model_type == "batch-top-k"
    assert m.expansion_factor == 4
    assert m.k == 64
    assert m.steps == 5000
    assert m.lr == 5e-4
    assert m.warmup_steps == 200
    assert m.config_tag() == "topk_x4_k64_lr5e-04_steps5000"
