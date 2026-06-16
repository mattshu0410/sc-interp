"""Hydra entry point for diffing.

Run:
    python -m scripts.diff method=pca pair=scgpt_base_vs_ft_norman capture=layer_11

Sweep:
    python -m scripts.diff -m method=pca,activation_diff capture=layer_0,layer_11
"""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import scripts.diffing.methods  # noqa: F401  triggers method registration
from scripts.diffing.base import get
from scripts.diffing.pairs import load_pair
from scripts.interp.hook_sinks import H5ActivationSink


def run(cfg: DictConfig) -> None:
    pair = load_pair(
        cfg.pair.a.path,
        cfg.pair.b.path,
        capture=cfg.capture.name,
        alignment=cfg.pair.alignment,
        relationship=cfg.pair.relationship,
        tags_a=dict(cfg.pair.a.get("tags", {})),
        tags_b=dict(cfg.pair.b.get("tags", {})),
    )

    method_cls = get(cfg.method.name)
    method = method_cls.from_config(cfg.method)

    tag = method.config_tag()
    out_dir = Path(cfg.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    with H5ActivationSink(
        out_dir / "scores.h5",
        runner=f"diff_{cfg.method.name}",
        dataset=cfg.pair.name,
        split="diff",
        capture_names=method_cls.output_capture_names,
        extra_meta={
            "method": cfg.method.name,
            "config_tag": tag,
            "pair_a_path": cfg.pair.a.path,
            "pair_b_path": cfg.pair.b.path,
            "pair_capture": cfg.capture.name,
            "alignment": cfg.pair.alignment,
            "relationship": cfg.pair.relationship,
            "config": OmegaConf.to_yaml(cfg),
        },
        mode=cfg.get("mode", "x"),
    ) as sink:
        method.fit(pair)
        method.score(pair, sink)

    method.save(out_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
