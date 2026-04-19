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


def test_empty_capture_names_rejected(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
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


def test_write_after_close_raises(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {}))
    with pytest.raises(RuntimeError, match="closed"):
        sink.write(_rec("lin", torch.zeros(1, 2), {}))


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


def test_h5_string_label_roundtrip(tmp_path: Path) -> None:
    # v3: per_cell may carry np.ndarray (dtype=object strings). Sink must
    # pick h5py.string_dtype for the dataset and append correctly across
    # batches.
    path = tmp_path / "act.h5"
    tags = {"phase": "predict"}
    b0_act = torch.randn(2, 4)
    b0_pert = np.array(["ctrl", "ETS2"], dtype=object)
    b1_act = torch.randn(3, 4)
    b1_pert = np.array(["CNN1", "ETS2+CNN1", "ctrl"], dtype=object)

    with _sink(path) as sink:
        sink.write(_rec("lin", b0_act, tags, per_cell={"pert": b0_pert}))
        sink.write(_rec("lin", b1_act, tags, per_cell={"pert": b1_pert}))

    with h5py.File(path, "r") as f:
        pert = f["lin/phase=predict/labels/pert"]
        # h5py reads vlen-utf8 datasets back with object dtype.
        assert pert.dtype == object
        assert pert.shape == (5,)
        got = [s.decode() if isinstance(s, bytes) else s for s in pert[...]]
        assert got == ["ctrl", "ETS2", "CNN1", "ETS2+CNN1", "ctrl"]


def test_h5_extra_meta_accepts_array_values(tmp_path: Path) -> None:
    # gene_symbols (array) and num_genes (int) must ride through extra_meta
    # so runners can make the h5 self-describing without touching the sink.
    path = tmp_path / "act.h5"
    genes = np.array(["TP53", "ETS2", "CNN1"], dtype=object)
    with H5ActivationSink(
        path,
        runner="scgpt",
        dataset="norman",
        split="test",
        capture_names=["lin"],
        extra_meta={"gene_symbols": genes, "num_genes": 3},
    ) as sink:
        sink.write(_rec("lin", torch.zeros(1, 2), {}))

    with h5py.File(path, "r") as f:
        # Arrays land under /meta/<k> as datasets (attr 64KB cap won't fit
        # real-world gene lists). Scalars stay as attrs.
        gs = f["meta/gene_symbols"][:]
        got = [s.decode() if isinstance(s, bytes) else s for s in gs]
        assert got == ["TP53", "ETS2", "CNN1"]
        assert int(f["meta"].attrs["num_genes"]) == 3


def test_h5_extra_meta_handles_large_string_array(tmp_path: Path) -> None:
    # Norman's gene_symbols (~5k strings) overflows the 64KB attr header;
    # the dataset path must accommodate that. 8000 strings is comfortably past.
    path = tmp_path / "act.h5"
    genes = np.array([f"GENE{i:05d}" for i in range(8000)], dtype=object)
    with H5ActivationSink(
        path,
        runner="gears",
        dataset="norman",
        split="test",
        capture_names=["gene_emb"],
        extra_meta={"gene_symbols": genes},
    ) as sink:
        sink.write(_rec("gene_emb", torch.zeros(1, 4), {}))

    with h5py.File(path, "r") as f:
        gs = f["meta/gene_symbols"][:]
        assert gs.shape == (8000,)
        first = gs[0].decode() if isinstance(gs[0], bytes) else gs[0]
        last = gs[-1].decode() if isinstance(gs[-1], bytes) else gs[-1]
        assert first == "GENE00000"
        assert last == "GENE07999"


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


def test_append_shape_mismatch_raises_and_preserves_dataset(tmp_path: Path) -> None:
    # h5py itself rejects a trailing-shape mismatch on resize/assign; we rely
    # on that instead of duplicating the check. Also verify the dataset was
    # not partially grown or corrupted.
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(2, 8), {}))
        with pytest.raises(Exception):
            sink.write(_rec("lin", torch.zeros(2, 16), {}))

    with h5py.File(path, "r") as f:
        assert f["lin/activation"].shape == (2, 8)


def test_append_dtype_mismatch_raises_and_preserves_dataset(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(2, 4, dtype=torch.float32), {}))
        with pytest.raises(ValueError, match="dtype mismatch"):
            sink.write(_rec("lin", torch.zeros(2, 4, dtype=torch.float16), {}))

    with h5py.File(path, "r") as f:
        dset = f["lin/activation"]
        assert dset.shape == (2, 4)
        assert dset.dtype == np.float32


def test_layout_attr_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(
            ActivationRecord(
                name="lin",
                tensor=torch.zeros(2, 4),
                metadata_tags={},
                layout="BTD",
            )
        )

    with h5py.File(path, "r") as f:
        assert f["lin/activation"].attrs["layout"] == "BTD"


def test_layout_unspecified_writes_no_attr(tmp_path: Path) -> None:
    # Empty-string layout means "not annotated" — don't pollute dataset attrs
    # with a layout key so consumers can distinguish absent from present.
    path = tmp_path / "act.h5"
    with _sink(path) as sink:
        sink.write(_rec("lin", torch.zeros(2, 4), {}))

    with h5py.File(path, "r") as f:
        assert "layout" not in f["lin/activation"].attrs


def test_per_cell_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="per_cell"):
        ActivationRecord(
            name="lin",
            tensor=torch.zeros(3, 4),
            metadata_tags={},
            per_cell={"cell_id": torch.arange(5)},
        )