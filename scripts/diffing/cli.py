"""CLI for `python -m scripts.diffing`."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.diffing.base import get, registered_methods
from scripts.diffing.pairs import load_pair
from scripts.interp.hook_sinks import H5ActivationSink
import scripts.diffing.methods  # noqa: F401  — triggers method registration


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m scripts.diffing",
        description="Compare two sets of cached activations.",
    )
    p.add_argument(
        "--method",
        required=True,
        choices=registered_methods(),
        help="Diffing method to run.",
    )
    p.add_argument("--a", required=True, type=Path, help="h5 path for model A activations.")
    p.add_argument("--b", required=True, type=Path, help="h5 path for model B activations.")
    p.add_argument(
        "--capture",
        required=True,
        help="Capture name (hook/layer) shared between A and B.",
    )
    p.add_argument("--out", required=True, type=Path, help="h5 path for score output.")
    p.add_argument(
        "--alignment",
        default="identity",
        help="Alignment strategy (identity | procrustes | cca | shared_sae). "
        "Only `identity` is implemented in iteration 1.",
    )
    p.add_argument(
        "--relationship",
        default="unspecified",
        choices=["finetune_vs_base", "cross_model", "checkpoint_steps", "unspecified"],
        help="How A and B relate — recorded as meta on the output file.",
    )
    p.add_argument(
        "--mode",
        default="x",
        choices=["x", "w"],
        help="Output file mode: 'x' refuses overwrite (default), 'w' truncates.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    pair = load_pair(
        args.a,
        args.b,
        capture=args.capture,
        alignment=args.alignment,
        relationship=args.relationship,
    )
    method_cls = get(args.method)
    method = method_cls()

    with H5ActivationSink(
        args.out,
        runner=f"diff_{args.method}",
        dataset=f"{args.a.stem}__vs__{args.b.stem}",
        split="diff",
        capture_names=method_cls.output_capture_names,
        extra_meta={
            "method": args.method,
            "pair_a_path": str(args.a),
            "pair_b_path": str(args.b),
            "pair_capture": args.capture,
            "alignment": args.alignment,
            "relationship": args.relationship,
        },
        mode=args.mode,
    ) as sink:
        method.fit(pair)
        method.score(pair, sink)
