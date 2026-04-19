"""Shared CLI flags for all runners.

common_parser returns an ArgumentParser preloaded with the flags every
runner needs: dataset, split, output, seed, force, hf-repo, and the
wandb flags. Runner-specific flags are added on top by the runner's
add_args callable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts import wb


def common_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Build an ArgumentParser with the flags every sc-interp runner uses."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--dataset", required=True, help="dataset name under data/")
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="which split to emit predictions for",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="prediction h5ad path; default predictions/<runner>_<dataset>_<split>.h5ad",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="ignore cached trained artifacts and retrain from scratch",
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HF Hub repo id for cached trained artifacts; pulled before training, pushed after",
    )
    p.add_argument(
        "--capture-activations",
        action="store_true",
        help="enable activation capture during predict (requires a runner wired for HookManager)",
    )
    p.add_argument(
        "--activation-out",
        type=Path,
        default=None,
        help="activation HDF5 path; default predictions/<runner>_<dataset>_<split>.activations.h5",
    )
    p.add_argument(
        "--capture-dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="dtype captured activations are cast to before sinking",
    )
    p.add_argument(
        "--capture-format",
        default="h5",
        choices=["h5", "memory"],
        help="activation sink format; memory is for tests/small notebooks only",
    )
    wb.add_args(p)
    return p
