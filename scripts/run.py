"""Runner dispatcher.

Usage:
    python -m scripts.run <runner> [runner args]

Example:
    python -m scripts.run scgpt --dataset norman --split test

The first positional argument selects a runner from REGISTRY. The
corresponding module is imported lazily so each runner's heavy,
venv-specific imports only run when that runner is actually invoked.
Register a new runner by adding a line to REGISTRY pointing at its
module path.
"""

from __future__ import annotations

import importlib
import sys

from scripts.runner import RunnerSpec, run

REGISTRY: dict[str, str] = {
    "scgpt": "scripts.run_scgpt",
    "cellflow": "scripts.run_cellflow",
    "gears": "scripts.run_gears",
}


def _load_spec(name: str) -> RunnerSpec:
    """Import the runner module and return its SPEC attribute."""
    module = importlib.import_module(REGISTRY[name])
    if not hasattr(module, "SPEC"):
        raise AttributeError(
            f"{REGISTRY[name]} has no SPEC attribute; every runner must export one"
        )
    return module.SPEC


def _print_top_help() -> None:
    """Print the dispatcher's own usage and the registered runner list."""
    print("usage: python -m scripts.run <runner> [runner args]")
    print()
    print("registered runners:", ", ".join(sorted(REGISTRY)))
    print()
    print("Pass --help after the runner name to see that runner's own flags.")


def main(argv: list[str] | None = None) -> None:
    """Split argv into <runner> + <runner-args> and dispatch."""
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        _print_top_help()
        return

    runner_name, rest = argv[0], argv[1:]
    if runner_name not in REGISTRY:
        print(f"unknown runner: {runner_name!r}")
        _print_top_help()
        sys.exit(2)

    spec = _load_spec(runner_name)
    run(spec, rest)


if __name__ == "__main__":
    main()
