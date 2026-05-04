from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.interp.hook_readers import H5ActivationReader
from scripts.interp.hook_sinks import H5ActivationSink, RunningStats
from scripts.interp.hooks import ActivationRecord


def _rec(
    name: str, tensor: torch.Tensor, tags: dict[str, str] | None = None,
) -> ActivationRecord:
    return ActivationRecord(
        name=name, tensor=tensor, metadata_tags=tags or {}, per_cell={},
    )


def _sink(path: Path, *, capture_names: list[str] | None = None, **kw: object) -> H5ActivationSink:
    return H5ActivationSink(
        path,
        runner="test",
        dataset="test",
        split="test",
        capture_names=capture_names or ["lin"],
        **kw,  # type: ignore[arg-type]
    )


# --- Welford math ---------------------------------------------------------


def test_running_stats_matches_torch_moments(tmp_path: Path) -> None:
    torch.manual_seed(0)
    n, d = 1000, 16
    full = torch.randn(n, d)
    batches = full.split(73, dim=0)

    path = tmp_path / "act.h5"
    with _sink(path, capture_names=["x"]) as sink:
        for t in batches:
            sink.write(_rec("x", t))

    with H5ActivationReader(path) as r:
        stats = r.running_stats("x")

    assert stats is not None
    assert stats.count == n
    torch.testing.assert_close(
        stats.mean.float(), full.mean(dim=0), atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(
        stats.std(unbiased=False).float(),
        full.std(dim=0, unbiased=False),
        atol=1e-5, rtol=1e-5,
    )
    torch.testing.assert_close(
        stats.std(unbiased=True).float(),
        full.std(dim=0, unbiased=True),
        atol=1e-5, rtol=1e-5,
    )


def test_running_stats_per_tag_isolation(tmp_path: Path) -> None:
    torch.manual_seed(1)
    a = torch.randn(50, 4) + 5.0
    b = torch.randn(50, 4) - 3.0

    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", a, {"phase": "train"}))
        sink.write(_rec("lin", b, {"phase": "eval"}))

    with H5ActivationReader(path) as r:
        stats_a = r.running_stats("lin", {"phase": "train"})
        stats_b = r.running_stats("lin", {"phase": "eval"})

    assert stats_a is not None and stats_b is not None
    torch.testing.assert_close(stats_a.mean.float(), a.mean(dim=0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stats_b.mean.float(), b.mean(dim=0), atol=1e-5, rtol=1e-5)


def test_running_stats_singleton_batches(tmp_path: Path) -> None:
    torch.manual_seed(2)
    rows = torch.randn(20, 6)

    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        for i in range(rows.shape[0]):
            sink.write(_rec("lin", rows[i : i + 1]))

    with H5ActivationReader(path) as r:
        stats = r.running_stats("lin")

    assert stats is not None and stats.count == 20
    torch.testing.assert_close(stats.mean.float(), rows.mean(dim=0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        stats.std(unbiased=False).float(),
        rows.std(dim=0, unbiased=False),
        atol=1e-5, rtol=1e-5,
    )


def test_running_stats_dtype_matches_input(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.randn(8, 4, dtype=torch.float32)))
    with H5ActivationReader(path) as r:
        stats = r.running_stats("lin")
    assert stats is not None
    assert stats.mean.dtype == torch.float32
    assert stats.M2.dtype == torch.float32


def test_running_stats_higher_rank_feature_shape(tmp_path: Path) -> None:
    # No layout annotation → preserve the (T, D) feature shape in stats.
    torch.manual_seed(5)
    full = torch.randn(80, 3, 5)
    batches = full.split(20, dim=0)

    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        for t in batches:
            sink.write(_rec("lin", t))

    with H5ActivationReader(path) as r:
        stats = r.running_stats("lin")

    assert stats is not None
    assert stats.mean.shape == (3, 5)
    assert stats.M2.shape == (3, 5)
    torch.testing.assert_close(stats.mean.float(), full.mean(dim=0), atol=1e-5, rtol=1e-5)


def test_running_stats_btd_collapses_token_axis(tmp_path: Path) -> None:
    torch.manual_seed(6)
    n_cells, t, d = 30, 7, 5
    full = torch.randn(n_cells, t, d)
    batches = full.split(10, dim=0)

    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        for batch in batches:
            sink.write(
                ActivationRecord(
                    name="lin", tensor=batch, metadata_tags={},
                    per_cell={}, layout="BTD",
                )
            )

    with H5ActivationReader(path) as r:
        stats = r.running_stats("lin")

    assert stats is not None
    assert stats.mean.shape == (d,)
    assert stats.M2.shape == (d,)
    assert stats.count == n_cells * t
    flat = full.reshape(-1, d)
    torch.testing.assert_close(stats.mean.float(), flat.mean(dim=0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        stats.std(unbiased=False).float(),
        flat.std(dim=0, unbiased=False),
        atol=1e-5, rtol=1e-5,
    )


# --- stats.h5 in folder mode ---------------------------------------------


def test_stats_h5_written_in_folder_mode(tmp_path: Path) -> None:
    torch.manual_seed(3)
    n_per_batch, d = 10, 8
    n_batches = 6
    batches = [torch.randn(n_per_batch, d) for _ in range(n_batches)]
    full = torch.cat(batches, dim=0)

    folder = tmp_path / "shards"
    with _sink(folder, capture_names=["x"], batches_per_shard=2) as sink:
        for t in batches:
            sink.write(_rec("x", t))
            sink.batch_end()

    shard_files = sorted(folder.glob("shard-*.h5"))
    assert len(shard_files) >= 2
    stats_file = folder / "stats.h5"
    assert stats_file.exists(), "folder mode should write stats.h5 sibling"

    # Per-shard files must NOT carry running_stats — single global blob lives in stats.h5.
    for sf in shard_files:
        with h5py.File(sf, "r") as f:
            assert "x/running_stats" not in f

    with H5ActivationReader(folder) as r:
        stats = r.running_stats("x")

    assert stats is not None
    assert stats.count == n_per_batch * n_batches
    torch.testing.assert_close(stats.mean.float(), full.mean(dim=0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        stats.std(unbiased=False).float(),
        full.std(dim=0, unbiased=False),
        atol=1e-5, rtol=1e-5,
    )


def test_stats_refreshed_on_each_shard_rotation(tmp_path: Path) -> None:
    # Mid-extraction crash should leave persisted stats consistent with the
    # rows already on disk: stats.h5 must be rewritten after every shard
    # rotation, not only on close.
    folder = tmp_path / "shards"
    sink = _sink(folder, capture_names=["x"], batches_per_shard=1)
    sink.__enter__()
    try:
        sink.write(_rec("x", torch.zeros(4, 3)))
        sink.batch_end()
        # After the first rotation the stats.h5 should already reflect the
        # one batch we wrote, even though we haven't closed the sink.
        assert (folder / "stats.h5").exists()
        with h5py.File(folder / "stats.h5", "r") as f:
            assert int(f["x/running_stats/count"][()]) == 4
        sink.write(_rec("x", torch.ones(4, 3)))
        sink.batch_end()
        with h5py.File(folder / "stats.h5", "r") as f:
            assert int(f["x/running_stats/count"][()]) == 8
    finally:
        sink.close()


def test_close_writes_trailing_batches(tmp_path: Path) -> None:
    # Last batch does not trigger a rotation; close() must still flush.
    folder = tmp_path / "shards"
    with _sink(folder, capture_names=["x"], batches_per_shard=10) as sink:
        sink.write(_rec("x", torch.randn(3, 2)))
        sink.batch_end()  # only 1 batch, no rotation
    assert (folder / "stats.h5").exists()
    with H5ActivationReader(folder) as r:
        stats = r.running_stats("x")
    assert stats is not None and stats.count == 3


# --- Parallel merge classmethod ------------------------------------------


def test_running_stats_parallel_merge_classmethod() -> None:
    torch.manual_seed(4)
    a = torch.randn(40, 5).double()
    b = torch.randn(60, 5).double()

    def _stats_of(t: torch.Tensor) -> RunningStats:
        m = t.mean(dim=0)
        return RunningStats(count=t.shape[0], mean=m, M2=((t - m) ** 2).sum(dim=0))

    merged = RunningStats.merge(_stats_of(a), _stats_of(b))
    full = torch.cat([a, b], dim=0)

    assert merged.count == 100
    torch.testing.assert_close(merged.mean, full.mean(dim=0), atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(
        merged.std(unbiased=False),
        full.std(dim=0, unbiased=False),
        atol=1e-10, rtol=1e-10,
    )


# --- Opt-out + legacy ----------------------------------------------------


def test_compute_running_stats_disabled_single_file(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path, compute_running_stats=False) as sink:
        sink.write(_rec("lin", torch.randn(10, 4)))

    with h5py.File(path, "r") as f:
        assert "lin/running_stats" not in f

    with H5ActivationReader(path) as r:
        assert r.running_stats("lin") is None


def test_compute_running_stats_disabled_folder(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    with _sink(folder, compute_running_stats=False, batches_per_shard=1) as sink:
        sink.write(_rec("lin", torch.randn(8, 3)))
        sink.batch_end()

    assert not (folder / "stats.h5").exists()
    with H5ActivationReader(folder) as r:
        assert r.running_stats("lin") is None


def test_legacy_file_returns_none(tmp_path: Path) -> None:
    # File written by older code (no running_stats sidecar): reader returns
    # None so callers can route to the backfill script.
    path = tmp_path / "legacy.h5"
    with _sink(path, compute_running_stats=False) as sink:
        sink.write(_rec("lin", torch.randn(8, 3)))

    with H5ActivationReader(path) as r:
        assert r.running_stats("lin") is None


# --- Backfill ------------------------------------------------------------


def test_backfill_single_file_matches_inline_stats(tmp_path: Path) -> None:
    from scripts.interp.backfill_running_stats import backfill

    torch.manual_seed(10)
    n, d = 200, 12
    full = torch.randn(n, d)
    batches = full.split(33, dim=0)

    inline = tmp_path / "inline.h5"
    legacy = tmp_path / "legacy.h5"

    with _sink(inline, capture_names=["x"]) as sink:
        for t in batches:
            sink.write(_rec("x", t))
    with _sink(legacy, capture_names=["x"], compute_running_stats=False) as sink:
        for t in batches:
            sink.write(_rec("x", t))

    n_groups = backfill(legacy)
    assert n_groups == 1

    with H5ActivationReader(inline) as r:
        inline_stats = r.running_stats("x")
    with H5ActivationReader(legacy) as r:
        legacy_stats = r.running_stats("x")

    assert inline_stats is not None and legacy_stats is not None
    assert inline_stats.count == legacy_stats.count
    torch.testing.assert_close(inline_stats.mean, legacy_stats.mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(inline_stats.M2, legacy_stats.M2, atol=1e-3, rtol=1e-4)


def test_backfill_folder_matches_inline_stats(tmp_path: Path) -> None:
    from scripts.interp.backfill_running_stats import backfill

    torch.manual_seed(11)
    n, d = 240, 6
    full = torch.randn(n, d)
    batches = full.split(20, dim=0)

    inline = tmp_path / "inline_shards"
    legacy = tmp_path / "legacy_shards"

    for folder, do_stats in [(inline, True), (legacy, False)]:
        with _sink(
            folder,
            capture_names=["x"],
            batches_per_shard=2,
            compute_running_stats=do_stats,
        ) as sink:
            for t in batches:
                sink.write(_rec("x", t))
                sink.batch_end()

    assert (inline / "stats.h5").exists()
    assert not (legacy / "stats.h5").exists()

    backfill(legacy)
    assert (legacy / "stats.h5").exists()

    with H5ActivationReader(inline) as r:
        inline_stats = r.running_stats("x")
    with H5ActivationReader(legacy) as r:
        legacy_stats = r.running_stats("x")

    assert inline_stats is not None and legacy_stats is not None
    assert inline_stats.count == legacy_stats.count == n
    torch.testing.assert_close(inline_stats.mean, legacy_stats.mean, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(inline_stats.M2, legacy_stats.M2, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(
        legacy_stats.mean.float(), full.mean(dim=0), atol=1e-5, rtol=1e-5,
    )


def test_backfill_captures_filter(tmp_path: Path) -> None:
    from scripts.interp.backfill_running_stats import backfill

    legacy = tmp_path / "legacy.h5"
    with _sink(legacy, capture_names=["lin_a", "lin_b"], compute_running_stats=False) as sink:
        sink.write(_rec("lin_a", torch.randn(10, 4)))
        sink.write(_rec("lin_b", torch.randn(10, 4)))

    n_groups = backfill(legacy, captures=["lin_a"])
    assert n_groups == 1

    with H5ActivationReader(legacy) as r:
        assert r.running_stats("lin_a") is not None
        assert r.running_stats("lin_b") is None


def test_backfill_honors_btd_layout(tmp_path: Path) -> None:
    from scripts.interp.backfill_running_stats import backfill

    torch.manual_seed(20)
    n_cells, t, d = 40, 6, 4
    full = torch.randn(n_cells, t, d)

    legacy = tmp_path / "legacy.h5"
    with _sink(legacy, compute_running_stats=False) as sink:
        sink.write(
            ActivationRecord(
                name="lin", tensor=full, metadata_tags={},
                per_cell={}, layout="BTD",
            )
        )

    backfill(legacy)

    with H5ActivationReader(legacy) as r:
        stats = r.running_stats("lin")

    assert stats is not None
    assert stats.mean.shape == (d,)
    assert stats.count == n_cells * t
    flat = full.reshape(-1, d)
    torch.testing.assert_close(stats.mean.float(), flat.mean(dim=0), atol=1e-5, rtol=1e-5)


def test_backfill_handles_tagged_groups(tmp_path: Path) -> None:
    from scripts.interp.backfill_running_stats import backfill

    a = torch.randn(50, 4) + 5.0
    b = torch.randn(50, 4) - 3.0

    legacy = tmp_path / "legacy.h5"
    with _sink(legacy, compute_running_stats=False) as sink:
        sink.write(_rec("lin", a, {"phase": "train"}))
        sink.write(_rec("lin", b, {"phase": "eval"}))

    n_groups = backfill(legacy)
    assert n_groups == 2

    with H5ActivationReader(legacy) as r:
        st_a = r.running_stats("lin", {"phase": "train"})
        st_b = r.running_stats("lin", {"phase": "eval"})

    assert st_a is not None and st_b is not None
    torch.testing.assert_close(st_a.mean.float(), a.mean(dim=0), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(st_b.mean.float(), b.mean(dim=0), atol=1e-5, rtol=1e-5)
