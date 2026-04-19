from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from scripts.interp.hook_readers import H5ActivationReader
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


def _write_sharded_fixture(
    folder: Path, batches_per_shard: int = 2, n_batches: int = 5
) -> list[torch.Tensor]:
    acts: list[torch.Tensor] = []
    torch.manual_seed(0)
    with H5ActivationSink(
        folder,
        runner="test",
        dataset="norman",
        split="test",
        capture_names=["blocks.0.attn"],
        batches_per_shard=batches_per_shard,
    ) as sink:
        for i in range(n_batches):
            a = torch.randn(3, 8) + i
            acts.append(a)
            sink.write(
                ActivationRecord(
                    name="blocks.0.attn",
                    tensor=a,
                    metadata_tags={"phase": "predict"},
                    per_cell={"cell_id": torch.arange(3) + i * 3},
                )
            )
            sink.batch_end()
    return acts


def test_reader_folder_equals_eager_concat_of_shards(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    acts = _write_sharded_fixture(folder, batches_per_shard=2, n_batches=5)
    # n_batches=5, batches_per_shard=2 → shards 0,1,2 (lazy rotation means
    # the trailing batch-end opens a third shard).
    assert len(sorted(folder.glob("shard-*.h5"))) == 3

    expected = torch.cat(acts, dim=0)
    with H5ActivationReader(folder) as r:
        act, labels = r.read("blocks.0.attn", {"phase": "predict"})
    assert torch.equal(act, expected)
    assert torch.equal(
        labels["cell_id"], torch.arange(15, dtype=labels["cell_id"].dtype)
    )


def test_reader_single_file_path_still_works(tmp_path: Path) -> None:
    # Regression: the folder auto-detect branch must not change single-file
    # behavior. Use the non-sharded sink, open the .h5 directly.
    path = tmp_path / "act.h5"
    torch.manual_seed(0)
    with H5ActivationSink(
        path,
        runner="test",
        dataset="norman",
        split="test",
        capture_names=["head"],
    ) as sink:
        for _ in range(3):
            sink.write(
                ActivationRecord(
                    name="head",
                    tensor=torch.randn(4, 6),
                    metadata_tags={"phase": "predict"},
                )
            )

    with H5ActivationReader(path) as r:
        act, _ = r.read("head", {"phase": "predict"})
        assert act.shape == (12, 6)
        assert r.meta["runner"] == "test"


def test_reader_rejects_shard_meta_mismatch(tmp_path: Path) -> None:
    # Build two shards whose /meta disagrees on `runner` — reader must
    # refuse at __enter__, not silently splice unrelated runs together.
    folder = tmp_path / "shards"
    folder.mkdir()

    for idx, runner in enumerate(["runner_a", "runner_b"]):
        with H5ActivationSink(
            folder / H5ActivationSink.SHARD_NAME_FORMAT.format(idx),
            runner=runner,
            dataset="d",
            split="s",
            capture_names=["lin"],
        ) as sink:
            sink.write(
                ActivationRecord(
                    name="lin",
                    tensor=torch.zeros(2, 4),
                    metadata_tags={},
                )
            )

    with pytest.raises(ValueError, match="shard meta mismatch"):
        H5ActivationReader(folder).__enter__()


def test_reader_rejects_shard_schema_version_mismatch(tmp_path: Path) -> None:
    # A lone shard written by an incompatible schema version must be
    # rejected the same way a single-file capture would be.
    folder = tmp_path / "shards"
    folder.mkdir()
    with h5py.File(folder / "shard-00000.h5", "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "99"
        meta.attrs["runner"] = "r"
        meta.attrs["dataset"] = "d"
        meta.attrs["split"] = "s"
        meta.attrs["capture_names"] = ["lin"]

    with pytest.raises(ValueError, match="schema_version"):
        H5ActivationReader(folder).__enter__()


def test_reader_rejects_empty_shard_folder(tmp_path: Path) -> None:
    folder = tmp_path / "shards"
    folder.mkdir()
    with pytest.raises(FileNotFoundError):
        H5ActivationReader(folder).__enter__()


def test_cross_shard_per_cell_labels_concat(tmp_path: Path) -> None:
    # Per-cell labels must concatenate along axis 0 in shard order, same
    # as the activation itself. Use a mix of torch tensor + object-dtype
    # (string) labels to hit both paths.
    folder = tmp_path / "shards"
    with H5ActivationSink(
        folder,
        runner="test",
        dataset="norman",
        split="test",
        capture_names=["lin"],
        batches_per_shard=1,
    ) as sink:
        for i in range(3):
            sink.write(
                ActivationRecord(
                    name="lin",
                    tensor=torch.full((2, 4), float(i)),
                    metadata_tags={"phase": "predict"},
                    per_cell={
                        "cell_id": torch.arange(2) + 2 * i,
                        "pert": np.array([f"p{i}a", f"p{i}b"], dtype=object),
                    },
                )
            )
            sink.batch_end()

    assert len(sorted(folder.glob("shard-*.h5"))) == 3

    with H5ActivationReader(folder) as r:
        act, labels = r.read("lin", {"phase": "predict"})
    assert act.shape == (6, 4)
    assert torch.equal(labels["cell_id"], torch.arange(6))
    assert list(labels["pert"]) == ["p0a", "p0b", "p1a", "p1b", "p2a", "p2b"]


def test_reader_skips_shard_missing_target(tmp_path: Path) -> None:
    # A gate that drops all records for one batch produces a shard with no
    # activation group. Reader must walk over it silently, not KeyError.
    folder = tmp_path / "shards"
    with H5ActivationSink(
        folder,
        runner="test",
        dataset="d",
        split="s",
        capture_names=["lin"],
        batches_per_shard=1,
    ) as sink:
        # Shard 0 gets a write.
        sink.write(
            ActivationRecord(
                name="lin",
                tensor=torch.full((2, 4), 1.0),
                metadata_tags={"phase": "predict"},
            )
        )
        sink.batch_end()
        # Shard 1: simulate a gate drop — batch_end fires with no writes.
        # The sink's lazy-open means shard-00001.h5 is never materialized;
        # this test asserts the reader survives that gap.
        sink.batch_end()
        # Shard 2 gets a write.
        sink.write(
            ActivationRecord(
                name="lin",
                tensor=torch.full((2, 4), 2.0),
                metadata_tags={"phase": "predict"},
            )
        )
        sink.batch_end()

    with H5ActivationReader(folder) as r:
        act, _ = r.read("lin", {"phase": "predict"})
    assert act.shape == (4, 4)
    assert torch.equal(
        act[:2], torch.full((2, 4), 1.0)
    ) and torch.equal(act[2:], torch.full((2, 4), 2.0))
