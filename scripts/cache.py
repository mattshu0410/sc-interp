"""Shared train-or-load cache flow for model runners.

Runners supply load/train/save callables and a cache layout; this module
handles the priority chain (local cache, HF Hub pull, train fresh) plus
training-stats persistence. TrainStats has a fixed top-level schema so
downstream consumers can read `uns["train_stats"]` without branching on
which runner produced it; model-specific extras live in `details`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from scripts import hf

T = TypeVar("T")


@dataclass
class TrainStats:
    """Fixed-schema training summary written alongside every cached model."""
    wall_clock_s: float = 0.0
    wandb_run_url: str | None = None
    reason: str = "trained"             # "trained" | "cached" | "early_stop" | "max_epochs" | "skipped"
    details: dict[str, Any] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Write to path as pretty-printed JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> TrainStats:
        """Read from path, assuming the current schema."""
        with open(path) as f:
            d = json.load(f)
        return cls(
            wall_clock_s=float(d.get("wall_clock_s", 0.0)),
            wandb_run_url=d.get("wandb_run_url"),
            reason=d.get("reason", "cached"),
            details=d.get("details", {}) or {},
        )


STATS_FILENAME = "training_stats.json"


def cache_or_train(
    *,
    cache_dir: Path,
    files: list[str],
    hf_repo: str | None,
    force: bool,
    load_cached: Callable[[Path], T],
    train: Callable[[], tuple[T, TrainStats]],
    save_trained: Callable[[T, Path], None],
) -> tuple[T, TrainStats]:
    """Resolve a trained model via local cache, HF Hub, or fresh training.

    Priority: force=True skips caches entirely. Otherwise, try local files
    first, then pull `files` from hf_repo into cache_dir if configured.
    Falls through to train() if neither hits. After training, writes the
    model via save_trained, persists TrainStats alongside it, and uploads
    to hf_repo if configured.

    `files` is the list of filenames cache_dir must contain to count as
    a cache hit. The stats file is handled internally and does not need
    to be listed.
    """
    stats_path = cache_dir / STATS_FILENAME

    def _cache_hit() -> bool:
        return all((cache_dir / f).exists() for f in files)

    if not force and not _cache_hit() and hf_repo:
        hf.try_download(hf_repo, cache_dir, files + [STATS_FILENAME])

    if not force and _cache_hit():
        print(f"==> cache hit at {cache_dir}, loading")
        model = load_cached(cache_dir)
        stats = (
            TrainStats.from_json(stats_path)
            if stats_path.exists()
            else TrainStats(reason="cached")
        )
        return model, stats

    model, stats = train()

    cache_dir.mkdir(parents=True, exist_ok=True)
    save_trained(model, cache_dir)
    stats.to_json(stats_path)
    print(f"==> saved trained artifacts to {cache_dir}")

    if hf_repo:
        hf.try_upload(hf_repo, cache_dir, files + [STATS_FILENAME])

    return model, stats
