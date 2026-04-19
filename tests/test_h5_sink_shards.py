from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


def _rec(
    name: str,
    tensor: torch.Tensor,
    tags: dict[str, str],
    per_cell: dict[str, torch.Tensor] | None = None,
) -> ActivationRecord:
    return ActivationRecord(
        name=name,
        tensor=tensor,
        metadata_tags=tags,
        per_cell=per_cell or {},
    )


def _sink(
    path: Path,
    capture_names: list[str] | None = None,
    **kwargs: object,
) -> H5ActivationSink:
    defaults: dict[str, object] = dict(
        runner="test",
        dataset="test",
        split="test",
        capture_names=capture_names or ["lin"],
    )
    defaults.update(kwargs)
    return H5ActivationSink(path, **defaults)  # type: ignore[arg-type]


def test_shard_rotation_produces_expected_file_count(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    with _sink(folder, batches_per_shard=2) as sink:
        for i in range(6):
            sink.write(_rec("lin", torch.zeros(3, 4), {"step": str(i)}))
            sink.batch_end()
    shards = sorted(folder.glob("shard-*.h5"))
    assert [p.name for p in shards] == [
        "shard-00000.h5",
        "shard-00001.h5",
        "shard-00002.h5",
    ]


def test_no_trailing_empty_shard(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    with _sink(folder, batches_per_shard=2) as sink:
        # Exactly N * batches_per_shard batches; final batch_end triggers
        # rotation but close() must not materialize the next (empty) shard.
        for i in range(4):
            sink.write(_rec("lin", torch.zeros(3, 4), {"step": str(i)}))
            sink.batch_end()
    shards = sorted(folder.glob("shard-*.h5"))
    assert [p.name for p in shards] == ["shard-00000.h5", "shard-00001.h5"]


def test_batch_end_is_noop_without_shard_setting(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(3, 4), {"phase": "predict"}))
        # Should simply return; no file rotation, no error.
        sink.batch_end()
        sink.batch_end()
        sink.write(_rec("lin", torch.zeros(2, 4), {"phase": "predict"}))
    with h5py.File(path, "r") as f:
        assert f["lin/phase=predict/activation"].shape == (5, 4)


def test_each_shard_carries_full_meta(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    with _sink(
        folder,
        capture_names=["lin", "other"],
        batches_per_shard=1,
        git_sha="cafe",
    ) as sink:
        for i in range(3):
            sink.write(_rec("lin", torch.zeros(2, 4), {"step": str(i)}))
            sink.batch_end()
    shards = sorted(folder.glob("shard-*.h5"))
    assert len(shards) == 3
    for shard in shards:
        with h5py.File(shard, "r") as f:
            meta = f["meta"].attrs
            assert meta["schema_version"] == H5ActivationSink.SCHEMA_VERSION
            assert meta["runner"] == "test"
            assert meta["dataset"] == "test"
            assert meta["split"] == "test"
            assert list(meta["capture_names"]) == ["lin", "other"]
            assert meta["git_sha"] == "cafe"


def test_mode_x_refuses_existing_folder(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    folder.mkdir()
    with pytest.raises(FileExistsError):
        _sink(folder, batches_per_shard=2, mode="x").__enter__()


def test_mode_w_wipes_prior_shards_not_unrelated_files(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    folder.mkdir()
    # Pre-existing shard file from a prior run; should be wiped.
    (folder / "shard-00000.h5").write_bytes(b"stale")
    # An unrelated file; the sink must refuse rather than clobber it.
    unrelated = folder / "README.txt"
    unrelated.write_text("keep me")

    with pytest.raises(FileExistsError):
        _sink(folder, batches_per_shard=1, mode="w").__enter__()
    assert unrelated.exists() and unrelated.read_text() == "keep me"
    # Prior shard still intact because the sink errored before wiping.
    assert (folder / "shard-00000.h5").read_bytes() == b"stale"

    # Remove the unrelated file, retry: prior shard should now be wiped and
    # replaced with a fresh one after one write+batch_end.
    unrelated.unlink()
    with _sink(folder, batches_per_shard=1, mode="w") as sink:
        sink.write(_rec("lin", torch.ones(2, 4), {"phase": "predict"}))
        sink.batch_end()
    shards = sorted(folder.glob("shard-*.h5"))
    assert [p.name for p in shards] == ["shard-00000.h5"]
    with h5py.File(shards[0], "r") as f:
        assert f["lin/phase=predict/activation"].shape == (2, 4)


def test_mode_x_refuses_when_path_is_existing_file(tmp_path: Path) -> None:
    # Sharded mode requires a folder-shaped target; a plain file at `path`
    # should fail fast rather than reinterpret it.
    path = tmp_path / "not-a-folder.h5"
    path.write_bytes(b"")
    with pytest.raises(NotADirectoryError):
        _sink(path, batches_per_shard=2).__enter__()


def test_write_after_rotation_lands_in_next_shard(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    with _sink(folder, batches_per_shard=1) as sink:
        sink.write(_rec("lin", torch.full((2, 4), 1.0), {"phase": "a"}))
        sink.batch_end()  # rotates; next write opens shard-00001.h5
        sink.write(_rec("lin", torch.full((2, 4), 2.0), {"phase": "b"}))
        sink.batch_end()

    with h5py.File(folder / "shard-00000.h5", "r") as f:
        assert "lin/phase=a/activation" in f
        assert "lin/phase=b/activation" not in f
    with h5py.File(folder / "shard-00001.h5", "r") as f:
        assert "lin/phase=b/activation" in f
        assert "lin/phase=a/activation" not in f


def test_batch_end_after_close_raises(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    sink = _sink(folder, batches_per_shard=2).__enter__()
    sink.close()
    with pytest.raises(RuntimeError):
        sink.batch_end()


def test_batches_per_shard_must_be_positive(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    with pytest.raises(ValueError):
        _sink(folder, batches_per_shard=0)
    with pytest.raises(ValueError):
        _sink(folder, batches_per_shard=-1)


def test_empty_batch_then_rotation_does_not_create_empty_shard(
    tmp_path: Path,
) -> None:
    # If a batch produces zero writes (e.g. gate dropped everything) AND
    # that batch_end triggers rotation, we must not leave a zero-record
    # shard behind — the next shard opens lazily on the next write.
    folder = tmp_path / "shards"
    with _sink(folder, batches_per_shard=1) as sink:
        sink.write(_rec("lin", torch.zeros(2, 4), {"phase": "a"}))
        sink.batch_end()  # closes shard-00000 with data
        sink.batch_end()  # would rotate to shard-00001; no writes between
        sink.write(_rec("lin", torch.zeros(2, 4), {"phase": "c"}))
        sink.batch_end()

    # Lazy file-open means the "empty" middle shard index is SKIPPED, not
    # written as an empty file. Result: two files on disk, with a gap in
    # the index sequence.
    shards = sorted(folder.glob("shard-*.h5"))
    assert [p.name for p in shards] == ["shard-00000.h5", "shard-00002.h5"]
