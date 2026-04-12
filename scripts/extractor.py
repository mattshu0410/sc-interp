"""ExtractSpec and the shared extract() orchestrator.

Parallel to runner.py / RunnerSpec, but for activation extraction rather
than perturbation prediction. Each extractor module exports a SPEC:
ExtractSpec describing its four steps (add_args, load_inputs, load_model,
extract). The extract() function handles CLI parsing, manifest loading, and
calls the four callables in order.

The output is NOT a predictions .h5ad — it is per-layer activation arrays
written to a HuggingFace dataset repo, with local memmaps and a CSV
progress file for resumability.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable

from scripts.cli import common_parser
from scripts.manifest import Manifest


@dataclass
class ExtractSpec:
    """Declarative extractor definition, dispatched by scripts.run."""
    name: str
    add_args: Callable[[argparse.ArgumentParser], None]
    load_inputs: Callable[[Manifest, argparse.Namespace], Any]
    load_model: Callable[[Any, argparse.Namespace], Any]
    extract: Callable[[Any, Any, argparse.Namespace], None]


def extract(spec: ExtractSpec, argv: list[str] | None = None) -> None:
    """Parse CLI, load manifest, run the spec's four steps in order."""
    parser = common_parser(description=f"sc-interp extractor: {spec.name}")
    spec.add_args(parser)
    args = parser.parse_args(argv)

    manifest = Manifest.load(args.dataset)
    print(f"==> extractor: {spec.name} | dataset: {manifest.name}")

    inputs = spec.load_inputs(manifest, args)
    model = spec.load_model(inputs, args)
    spec.extract(model, inputs, args)
