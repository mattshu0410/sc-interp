from __future__ import annotations

from pathlib import Path

import pytest
import torch

from scripts.interp.hook_readers import H5ActivationReader
from scripts.interp.hook_sinks import H5ActivationSink
from scripts.interp.hooks import ActivationRecord


def _write_fixture(path: Path) -> None:
    with H5ActivationSink(
        path,
        runner="test",
        dataset="norman",
        split="test",
        capture_names=["blocks.0.attn", "head"],
        git_sha="deadbeef",
    ) as sink:
        torch.manual_seed(0)
        for i in range(2):
            act = torch.randn(3, 8) + i
            cid = torch.arange(3) + 10 * i
            sink.write(
                ActivationRecord(
                    name="blocks.0.attn",
                    tensor=act,
                    metadata_tags={"phase": "predict"},
                    per_cell={"cell_id": cid},
                )
            )
        sink.write(
            ActivationRecord(
                name="head",
                tensor=torch.zeros(2, 4),
                metadata_tags={},
            )
        )


def test_reader_meta_exposes_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    _write_fixture(path)
    with H5ActivationReader(path) as r:
        m = r.meta
        assert m["runner"] == "test"
        assert m["dataset"] == "norman"
        assert m["split"] == "test"
        assert r.capture_names() == ["blocks.0.attn", "head"]
        assert m["git_sha"] == "deadbeef"


def test_reader_round_trip_activation_and_labels(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    _write_fixture(path)

    # Reproduce the fixture's writes to compare against.
    torch.manual_seed(0)
    expected_acts = torch.cat([torch.randn(3, 8) + i for i in range(2)])

    with H5ActivationReader(path) as r:
        act, labels = r.read("blocks.0.attn", {"phase": "predict"})
        assert torch.equal(act, expected_acts)
        assert set(labels) == {"cell_id"}
        assert torch.equal(labels["cell_id"], torch.tensor([0, 1, 2, 10, 11, 12]))


def test_reader_handles_no_tags_and_no_labels(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    _write_fixture(path)
    with H5ActivationReader(path) as r:
        act, labels = r.read("head")
        assert act.shape == (2, 4)
        assert labels == {}


def test_reader_missing_capture_raises(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    _write_fixture(path)
    with H5ActivationReader(path) as r:
        with pytest.raises(KeyError):
            r.read("blocks.17.attn", {})


def test_reader_layout_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with H5ActivationSink(
        path, runner="r", dataset="d", split="s", capture_names=["a", "b"]
    ) as sink:
        sink.write(
            ActivationRecord(
                name="a",
                tensor=torch.zeros(2, 3, 4),
                metadata_tags={},
                layout="BTD",
            )
        )
        sink.write(
            ActivationRecord(
                name="b",
                tensor=torch.zeros(2, 4),
                metadata_tags={},
                # layout defaults to ""
            )
        )

    with H5ActivationReader(path) as r:
        assert r.layout("a") == "BTD"
        assert r.layout("b") == ""


def test_reader_rejects_mismatched_schema(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "act.h5"
    with h5py.File(path, "w") as f:
        f.create_group("meta").attrs["schema_version"] = "99"
    with pytest.raises(ValueError, match="schema_version"):
        with H5ActivationReader(path):
            pass
