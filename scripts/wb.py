"""WandB wrapper for model runners.

Exposes add_args/init/log/finish/url. No-ops silently when wandb is not
installed or --wandb-mode=disabled.
"""

import argparse
from typing import Any

_run = None  # module-level handle to the active run


def add_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("wandb")
    g.add_argument(
        "--wandb-project",
        default="sc-interp",
        help="wandb project name, default sc-interp",
    )
    g.add_argument(
        "--wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb mode, online requires login",
    )
    g.add_argument(
        "--wandb-tags",
        default=None,
        help="comma-separated extra tags to attach to the run",
    )
    g.add_argument(
        "--wandb-name",
        default=None,
        help="override the auto-generated run name",
    )


def init(
    model: str,
    dataset: str,
    args: argparse.Namespace,
    extra_config: dict[str, Any] | None = None,
) -> None:
    """Start a wandb run for this training invocation.

    No-op if --wandb-mode=disabled or wandb is not installed. The model
    name and dataset are recorded as both tags and config fields so runs
    can be filtered on the dashboard.
    """
    global _run
    if getattr(args, "wandb_mode", "disabled") == "disabled":
        return
    try:
        import wandb
    except ImportError:
        print("==> wandb not installed, logging disabled")
        return

    tags = [model, dataset]
    if getattr(args, "wandb_tags", None):
        tags.extend(t.strip() for t in args.wandb_tags.split(",") if t.strip())

    config = {**vars(args), "model": model, "dataset": dataset}
    if extra_config:
        config.update(extra_config)

    _run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name or f"{model}-{dataset}",
        config=config,
        tags=tags,
        mode=args.wandb_mode,
    )
    print(f"==> wandb run: {_run.url}")


def log(metrics: dict[str, Any], step: int | None = None) -> None:
    """Log a metrics dict to the active wandb run. No-op if disabled."""
    if _run is None:
        return
    import wandb
    wandb.log(metrics, step=step)


def finish() -> None:
    """Close the active wandb run. Safe to call even if init was skipped."""
    global _run
    if _run is None:
        return
    import wandb
    wandb.finish()
    _run = None


def url() -> str | None:
    """Return the URL of the active run, or None if wandb is disabled."""
    return _run.url if _run is not None else None
