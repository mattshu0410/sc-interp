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


def _sink(path: Path, capture_names: list[str] | None = None, **kwargs: object) -> H5ActivationSink:
    defaults: dict[str, object] = dict(
        runner="test",
        dataset="test",
        split="test",
        capture_names=capture_names or ["lin"],
    )
    defaults.update(kwargs)
    return H5ActivationSink(path, **defaults)  # type: ignore[arg-type]


def test_append_same_key_grows_axis_0(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    torch.manual_seed(0)
    batches = [torch.randn(5, 8) for _ in range(4)]
    tags = {"phase": "predict", "step": "0"}

    with _sink(path, capture_names=["blocks.0.attn"]) as sink:
        for t in batches:
            sink.write(_rec("blocks.0.attn", t, tags))

    with h5py.File(path, "r") as f:
        dset = f["blocks.0.attn/phase=predict/step=0/activation"]
        assert dset.shape == (20, 8)
        assert dset.dtype == np.float32
        assert dset.chunks is not None
        expected = torch.cat(batches).numpy()
        assert np.array_equal(dset[...], expected)
        assert dset.attrs["name"] == "blocks.0.attn"
        assert dset.attrs["phase"] == "predict"
        assert dset.attrs["step"] == "0"
        assert f["meta"].attrs["schema_version"] == H5ActivationSink.SCHEMA_VERSION


def test_distinct_tags_create_distinct_datasets(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    t0 = torch.randn(3, 4)
    t1 = torch.randn(3, 4)

    with _sink(path) as sink:
        sink.write(_rec("lin", t0, {"step": "0"}))
        sink.write(_rec("lin", t1, {"step": "1"}))

    with h5py.File(path, "r") as f:
        d0 = f["lin/step=0/activation"]
        d1 = f["lin/step=1/activation"]
        assert np.array_equal(d0[...], t0.numpy())
        assert np.array_equal(d1[...], t1.numpy())
        assert d0.attrs["step"] == "0"
        assert d1.attrs["step"] == "1"


def test_meta_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with H5ActivationSink(
        path,
        runner="scgpt",
        dataset="norman",
        split="test",
        capture_names=["lin", "blocks.0.attn"],
        git_sha="abc123",
        extra_meta={"notes": "smoke"},
    ) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {}))

    with h5py.File(path, "r") as f:
        m = f["meta"].attrs
        assert m["runner"] == "scgpt"
        assert m["dataset"] == "norman"
        assert m["split"] == "test"
        assert list(m["capture_names"]) == ["lin", "blocks.0.attn"]
        assert m["git_sha"] == "abc123"
        assert m["notes"] == "smoke"
        assert m["schema_version"] == H5ActivationSink.SCHEMA_VERSION


def test_required_meta_enforced(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with pytest.raises(TypeError):
        H5ActivationSink(path)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="capture_names"):
        H5ActivationSink(
            path, runner="r", dataset="d", split="s", capture_names=[]
        )


def test_existing_file_refused_by_default(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {}))
    with pytest.raises(FileExistsError):
        with _sink(path) as _:
            pass


def test_mode_w_allows_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.ones(1, 2), {}))
    with _sink(path, mode="w") as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {}))

    with h5py.File(path, "r") as f:
        assert np.array_equal(f["lin/activation"][...], np.zeros((1, 2), dtype=np.float32))


def test_extra_meta_cannot_override_reserved(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with pytest.raises(ValueError, match="reserved"):
        with H5ActivationSink(
            path,
            runner="r",
            dataset="d",
            split="s",
            capture_names=["lin"],
            extra_meta={"runner": "other"},
        ):
            pass


def test_dtype_preserved_fp16_and_fp32(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    t_fp32 = torch.randn(2, 4, dtype=torch.float32)
    t_fp16 = torch.randn(2, 4, dtype=torch.float16)

    with _sink(path) as sink:
        sink.write(_rec("fp32_layer", t_fp32, {"dtype": "fp32"}))
        sink.write(_rec("fp16_layer", t_fp16, {"dtype": "fp16"}))

    with h5py.File(path, "r") as f:
        assert f["fp32_layer/dtype=fp32/activation"].dtype == np.float32
        assert f["fp16_layer/dtype=fp16/activation"].dtype == np.float16
        assert np.array_equal(f["fp32_layer/dtype=fp32/activation"][...], t_fp32.numpy())
        assert np.array_equal(f["fp16_layer/dtype=fp16/activation"][...], t_fp16.numpy())


def test_tag_path_is_sorted_deterministically(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    tags_a = {"b": "2", "a": "1"}
    tags_b = {"a": "1", "b": "2"}  # same content, different insertion order

    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), tags_a))
        sink.write(_rec("lin", torch.ones(1, 2), tags_b))

    with h5py.File(path, "r") as f:
        dset = f["lin/a=1/b=2/activation"]
        assert dset.shape == (2, 2)


def test_write_after_close_raises(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {}))
    with pytest.raises(RuntimeError, match="closed"):
        sink.write(_rec("lin", torch.zeros(1, 2), {}))


def test_no_tags_places_dataset_at_name_root(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(3, 4), {}))

    with h5py.File(path, "r") as f:
        assert f["lin/activation"].shape == (3, 4)


def test_gzip_compression_applied(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(100, 32), {}))

    with h5py.File(path, "r") as f:
        dset = f["lin/activation"]
        assert dset.compression == "gzip"
        assert dset.compression_opts == 4


def test_per_cell_labels_written_and_aligned(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    tags = {"phase": "predict"}
    torch.manual_seed(0)
    batches = [
        (torch.randn(3, 4), torch.arange(3) + 10 * i, torch.full((3,), i, dtype=torch.int32))
        for i in range(2)
    ]

    with _sink(path) as sink:
        for act, cell_id, batch_idx in batches:
            sink.write(
                _rec(
                    "lin",
                    act,
                    tags,
                    per_cell={"cell_id": cell_id, "batch_idx": batch_idx},
                )
            )

    with h5py.File(path, "r") as f:
        act = f["lin/phase=predict/activation"]
        cid = f["lin/phase=predict/labels/cell_id"]
        bidx = f["lin/phase=predict/labels/batch_idx"]
        assert act.shape == (6, 4)
        assert cid.shape == (6,)
        assert bidx.shape == (6,)
        expected_cid = torch.cat([b[1] for b in batches]).numpy()
        expected_bidx = torch.cat([b[2] for b in batches]).numpy()
        assert np.array_equal(cid[...], expected_cid)
        assert np.array_equal(bidx[...], expected_bidx)


def test_path_escaping_prevents_collision(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    # A tag value containing '/' would, without escaping, create nested groups
    # that collide with a different tag's structure.
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {"epoch": "1/10"}))
        sink.write(_rec("lin", torch.ones(1, 2), {"epoch": "1", "step": "10"}))

    with h5py.File(path, "r") as f:
        d0 = f["lin/epoch=1%2F10/activation"]
        d1 = f["lin/epoch=1/step=10/activation"]
        assert d0.shape == (1, 2)
        assert d1.shape == (1, 2)
        assert np.array_equal(d0[...], np.zeros((1, 2), dtype=np.float32))
        assert np.array_equal(d1[...], np.ones((1, 2), dtype=np.float32))


def test_path_escaping_handles_equals_and_percent(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {"cfg": "a=b%c"}))

    with h5py.File(path, "r") as f:
        assert "lin/cfg=a%3Db%25c/activation" in f


def test_append_shape_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(2, 8), {}))
        with pytest.raises(ValueError, match="shape mismatch"):
            sink.write(_rec("lin", torch.zeros(2, 16), {}))


def test_append_dtype_mismatch_raises(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(2, 4, dtype=torch.float32), {}))
        with pytest.raises(ValueError, match="dtype mismatch"):
            sink.write(_rec("lin", torch.zeros(2, 4, dtype=torch.float16), {}))


def test_per_cell_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="per_cell"):
        ActivationRecord(
            name="lin",
            tensor=torch.zeros(3, 4),
            metadata_tags={},
            per_cell={"cell_id": torch.arange(5)},
        )