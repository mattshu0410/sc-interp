from __future__ import annotations

from pathlib import Path

import h5py

from scripts.interp.hooks import ActivationRecord


def _escape_segment(s: str) -> str:
    # Percent-encode characters that carry structural meaning in the h5
    # tag-subpath schema: '/' (group separator), '=' (key/value delimiter
    # inside a tag segment), and '%' itself (so the encoding is reversible).
    # Without this, `{"layer": "1/2"}` or `name="blocks/0"` silently create
    # ambiguous nested groups that collide with legit names.
    return s.replace("%", "%25").replace("/", "%2F").replace("=", "%3D")


def group_path(name: str, tags: dict[str, str]) -> str:
    esc_name = _escape_segment(name)
    segments = [
        f"{_escape_segment(k)}={_escape_segment(v)}" for k, v in sorted(tags.items())
    ]
    if segments:
        return f"/{esc_name}/" + "/".join(segments)
    return f"/{esc_name}"


class MemoryActivationSink:
    def __init__(self) -> None:
        self.records: list[ActivationRecord] = []
        self._closed = False

    def write(self, record: ActivationRecord) -> None:
        if self._closed:
            raise RuntimeError("write() called on closed MemoryActivationSink")
        self.records.append(record)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> MemoryActivationSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class H5ActivationSink:
    SCHEMA_VERSION = "2"
    REQUIRED_META_FIELDS = ("runner", "dataset", "split", "capture_names")

    def __init__(
        self,
        path: str | Path,
        *,
        runner: str,
        dataset: str,
        split: str,
        capture_names: list[str],
        git_sha: str | None = None,
        extra_meta: dict[str, str] | None = None,
        compression_opts: int = 4,
        mode: str = "x",
    ) -> None:
        if not capture_names:
            raise ValueError("capture_names must be a non-empty list")
        if mode not in ("x", "w"):
            raise ValueError(
                f"mode must be 'x' (refuse overwrite, default) or 'w' "
                f"(truncate), got {mode!r}"
            )
        self.path = Path(path)
        self.runner = runner
        self.dataset = dataset
        self.split = split
        self.capture_names = list(capture_names)
        self.git_sha = git_sha
        self.extra_meta = dict(extra_meta or {})
        self.compression_opts = compression_opts
        self.mode = mode
        self._file: h5py.File | None = None
        self._closed = False

    def __enter__(self) -> H5ActivationSink:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "x" and self.path.exists():
            raise FileExistsError(
                f"{self.path} already exists; pass mode='w' to overwrite"
            )
        self._file = h5py.File(self.path, "w")
        meta_grp = self._file.create_group("meta")
        meta_grp.attrs["schema_version"] = self.SCHEMA_VERSION
        meta_grp.attrs["runner"] = self.runner
        meta_grp.attrs["dataset"] = self.dataset
        meta_grp.attrs["split"] = self.split
        meta_grp.attrs["capture_names"] = self.capture_names
        if self.git_sha is not None:
            meta_grp.attrs["git_sha"] = self.git_sha
        for k, v in self.extra_meta.items():
            if k in self.REQUIRED_META_FIELDS or k in ("schema_version", "git_sha"):
                raise ValueError(
                    f"extra_meta may not override reserved field {k!r}"
                )
            meta_grp.attrs[k] = v
        return self

    def write(self, record: ActivationRecord) -> None:
        if self._closed or self._file is None:
            raise RuntimeError("write() called on closed H5ActivationSink")
        gpath = group_path(record.name, record.metadata_tags)
        arr = record.tensor.detach().cpu().numpy()
        act_path = f"{gpath}/activation"
        if act_path in self._file:
            dset = self._file[act_path]
            if dset.dtype != arr.dtype:
                raise ValueError(
                    f"H5 append dtype mismatch at {act_path}: existing "
                    f"{dset.dtype} != new {arr.dtype}"
                )
            old = dset.shape[0]
            new_shape = (old + arr.shape[0],) + arr.shape[1:]
            dset.resize(new_shape)
            dset[old:] = arr
        else:
            maxshape = (None,) + arr.shape[1:]
            dset = self._file.create_dataset(
                act_path,
                data=arr,
                maxshape=maxshape,
                chunks=True,
                compression="gzip",
                compression_opts=self.compression_opts,
            )
            dset.attrs["name"] = record.name
            if record.layout:
                dset.attrs["layout"] = record.layout
            for k, v in record.metadata_tags.items():
                dset.attrs[k] = v

        for label_name, label_tensor in record.per_cell.items():
            label_arr = label_tensor.detach().cpu().numpy()
            label_path = f"{gpath}/labels/{label_name}"
            if label_path in self._file:
                ldset = self._file[label_path]
                if ldset.dtype != label_arr.dtype:
                    raise ValueError(
                        f"H5 append dtype mismatch at {label_path}: existing "
                        f"{ldset.dtype} != new {label_arr.dtype}"
                    )
                old = ldset.shape[0]
                ldset.resize((old + label_arr.shape[0],) + label_arr.shape[1:])
                ldset[old:] = label_arr
            else:
                lmaxshape = (None,) + label_arr.shape[1:]
                self._file.create_dataset(
                    label_path,
                    data=label_arr,
                    maxshape=lmaxshape,
                    chunks=True,
                    compression="gzip",
                    compression_opts=self.compression_opts,
                )

    def close(self) -> None:
        if self._closed:
            return
        if self._file is not None:
            self._file.close()
            self._file = None
        self._closed = True

    def __exit__(self, *exc: object) -> None:
        self.close()
