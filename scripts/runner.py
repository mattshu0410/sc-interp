"""RunnerSpec and the shared run() orchestrator.

Each runner module exports a SPEC: RunnerSpec describing its five steps
(add_args, load_inputs, train_or_load, predict, save_predictions). The
run() function parses CLI, loads the manifest, inits wandb, and calls
the five callables in order. Runner modules own the step implementations
but not the outer flow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from scripts import wb
from scripts.cache import TrainStats
from scripts.cli import common_parser
from scripts.manifest import Manifest

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunnerSpec:
    """Declarative runner definition, dispatched by scripts.run."""
    name: str
    add_args: Callable[[argparse.ArgumentParser], None]
    load_inputs: Callable[[Manifest, argparse.Namespace], Any]
    train_or_load: Callable[[Any, str, argparse.Namespace], tuple[Any, TrainStats]]
    predict: Callable[[Any, Any, argparse.Namespace], Any]
    save_predictions: Callable[
        [Any, Any, Manifest, argparse.Namespace, TrainStats, Path], None
    ]


def default_output(runner_name: str, dataset: str, split: str) -> Path:
    """predictions/<runner>_<dataset>_<split>.h5ad under the repo root."""
    return REPO_ROOT / "predictions" / f"{runner_name}_{dataset}_{split}.h5ad"


def run(spec: RunnerSpec, argv: list[str] | None = None) -> None:
    """Parse CLI, load manifest, run the spec's five steps in order."""
    parser = common_parser(description=f"sc-interp runner: {spec.name}")
    spec.add_args(parser)
    args = parser.parse_args(argv)

    manifest = Manifest.load(args.dataset)
    print(f"==> runner: {spec.name} | dataset: {manifest.name}")

    wb.init(spec.name, args.dataset, args)
    try:
        inputs = spec.load_inputs(manifest, args)
        model, stats = spec.train_or_load(inputs, args.dataset, args)
        preds = spec.predict(model, inputs, args)
        output = args.output or default_output(spec.name, args.dataset, args.split)
        spec.save_predictions(preds, inputs, manifest, args, stats, output)
    finally:
        wb.finish()
