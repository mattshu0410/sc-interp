from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from scripts.diff import run
from scripts.interp.hook_readers import H5ActivationReader
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
CAPTURE = "layer_0"


def _write(path: Path, tensor: torch.Tensor, labels: dict | None = None) -> None:
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
                metadata_tags={"phase": "predict"},
                per_cell=labels or {},
                layout="BD",
            )
        )


@pytest.fixture
def synthetic_pair(tmp_path: Path) -> tuple[Path, Path]:
    torch.manual_seed(0)
    n, d = 256, 16
    a = torch.randn(n, d)
    b = a + torch.randn(n, d) * 0.1
    path_a = tmp_path / "a.h5"
    path_b = tmp_path / "b.h5"
    _write(path_a, a, labels={"cell_id": np.arange(n)})
    _write(path_b, b)
    return path_a, path_b


def _compose(method: str, path_a: Path, path_b: Path, out_dir: Path, **extra):
    overrides = [
        f"method={method}",
        "pair=scgpt_base_vs_ft_norman",
        f"capture={CAPTURE}",
        # Synthetic data uses capture key "layer_0" instead of the production
        # scGPT key; tags match production (phase=predict).
        f"capture.name={CAPTURE}",
        f"pair.a.path={path_a}",
        f"pair.b.path={path_b}",
        f"out_dir={out_dir}",
    ]
    overrides.extend(f"{k}={v}" for k, v in extra.items())
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=overrides)


def test_hydra_runs_activation_diff(synthetic_pair, tmp_path):
    path_a, path_b = synthetic_pair
    out_dir = tmp_path / "out"
    cfg = _compose("activation_diff", path_a, path_b, out_dir)
    run(cfg)

    scores = out_dir / "default" / "scores.h5"
    with H5ActivationReader(scores) as r:
        l2, labels = r.read("l2_per_cell")
        cosine, _ = r.read("cosine_per_cell")
        meta = r.meta
    assert l2.shape == (256,)
    assert cosine.shape == (256,)
    assert "cell_id" in labels
    assert meta["runner"] == "diff_activation_diff"
    assert meta["dataset"] == "scgpt_base_vs_ft_norman"
    assert meta["config_tag"] == "default"


def test_hydra_runs_pca(synthetic_pair, tmp_path):
    path_a, path_b = synthetic_pair
    out_dir = tmp_path / "out"
    cfg = _compose(
        "pca", path_a, path_b, out_dir,
        **{"method.training.n_components": 4, "method.training.batch_size": 64},
    )
    run(cfg)

    scores = out_dir / "n4_b_minus_a" / "scores.h5"
    with H5ActivationReader(scores) as r:
        pc_scores, _ = r.read("pc_scores")
        evr, _ = r.read("explained_variance_ratio")
    assert pc_scores.shape == (256, 4)
    assert evr.shape == (1, 4)
    assert (out_dir / "n4_b_minus_a" / "pca_model.pkl").exists()


def test_hydra_method_override(synthetic_pair, tmp_path):
    path_a, path_b = synthetic_pair
    out_dir = tmp_path / "out"
    cfg = _compose(
        "pca", path_a, path_b, out_dir,
        **{"method.training.n_components": 8, "method.training.target": "a_minus_b"},
    )
    assert cfg.method.training.n_components == 8
    assert cfg.method.training.target == "a_minus_b"
    run(cfg)
    scores = out_dir / "n8_a_minus_b" / "scores.h5"
    with H5ActivationReader(scores) as r:
        pc_scores, _ = r.read("pc_scores")
    assert pc_scores.shape == (256, 8)


def test_config_tag_namespaces_reruns(synthetic_pair, tmp_path):
    path_a, path_b = synthetic_pair
    out_dir = tmp_path / "out"
    run(_compose(
        "pca", path_a, path_b, out_dir,
        **{"method.training.n_components": 4, "method.training.batch_size": 64},
    ))
    run(_compose(
        "pca", path_a, path_b, out_dir,
        **{"method.training.n_components": 8, "method.training.batch_size": 64},
    ))
    assert (out_dir / "n4_b_minus_a" / "scores.h5").exists()
    assert (out_dir / "n8_b_minus_a" / "scores.h5").exists()
