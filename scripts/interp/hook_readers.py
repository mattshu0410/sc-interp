from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from scripts.interp.hook_sinks import H5ActivationSink, RunningStats, group_path
from scripts.interp.hooks import Layout  # re-exported for callers


_META_CONSISTENCY_FIELDS = (
    "schema_version", "runner", "dataset", "split", "capture_names",
)


class H5ActivationReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # _files is the ordered shard list (length 1 in single-file mode).
        # Reader logic always iterates _files, so single-file is just the
        # degenerate shard case — no branchy code paths.
        self._files: list[h5py.File] = []
        # Folder-mode sibling stats file; None when absent or single-file.
        self._stats_file: h5py.File | None = None

    def __enter__(self) -> H5ActivationReader:
        if self.path.is_dir():
            shard_paths = sorted(self.path.glob(H5ActivationSink._SHARD_GLOB))
            if not shard_paths:
                raise FileNotFoundError(
                    f"{self.path} is a directory but contains no "
                    f"{H5ActivationSink._SHARD_GLOB} files"
                )
            self._files = [h5py.File(p, "r") for p in shard_paths]
            stats_path = self.path / H5ActivationSink.STATS_FILE_NAME
            if stats_path.exists():
                self._stats_file = h5py.File(stats_path, "r")
        else:
            self._files = [h5py.File(self.path, "r")]
        self._validate_shards()
        return self

    def _validate_shards(self) -> None:
        # Each shard is independently openable. Mismatched meta means the
        # folder was assembled from two unrelated captures — refuse rather
        # than silently concatenate rows that don't belong together.
        expected_version = H5ActivationSink.SCHEMA_VERSION
        for f in self._files:
            version = f["meta"].attrs.get("schema_version")
            if isinstance(version, bytes):
                version = version.decode()
            if version != expected_version:
                raise ValueError(
                    f"schema_version {version!r} at {f.filename} does not "
                    f"match reader expected {expected_version!r}"
                )
        if len(self._files) > 1:
            first = self._files[0]["meta"].attrs
            for f in self._files[1:]:
                for field in _META_CONSISTENCY_FIELDS:
                    a, b = first.get(field), f["meta"].attrs.get(field)
                    if not _attrs_equal(a, b):
                        raise ValueError(
                            f"shard meta mismatch at {f.filename}: "
                            f"{field}={b!r} != first shard's {a!r}"
                        )

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files = []
        if self._stats_file is not None:
            self._stats_file.close()
            self._stats_file = None

    @property
    def meta(self) -> dict[str, Any]:
        self._require_open()
        # Meta is replicated across shards and validated to be equal at
        # __enter__; the first shard is the authoritative view.
        out: dict[str, Any] = {}
        for k, v in self._files[0]["meta"].attrs.items():
            if isinstance(v, bytes):
                out[k] = v.decode()
            elif isinstance(v, np.ndarray):
                out[k] = [
                    x.decode() if isinstance(x, bytes) else str(x) for x in v
                ]
            else:
                out[k] = v
        return out

    def capture_names(self) -> list[str]:
        names = self.meta.get("capture_names", [])
        if isinstance(names, list):
            return names
        return [str(names)]

    def layout(self, name: str, tags: dict[str, str] | None = None) -> Layout:
        self._require_open()
        gpath = group_path(name, tags or {})
        act_path = f"{gpath}/activation"
        # First shard containing this dataset wins — the layout annotation
        # is a per-(name, tags) invariant, so shard 0 suffices once we've
        # validated meta consistency.
        for f in self._files:
            if act_path in f:
                attr = f[act_path].attrs.get("layout", "")
                if isinstance(attr, bytes):
                    attr = attr.decode()
                return str(attr)  # type: ignore[return-value]
        raise KeyError(f"no activation at {act_path}")

    def running_stats(
        self, name: str, tags: dict[str, str] | None = None
    ) -> RunningStats | None:
        """Welford stats for one (capture, tags) group.

        Folder mode reads `<folder>/stats.h5`; single-file mode reads from
        the activation file. Returns None if the sidecar is absent (e.g.
        captures written by older code; run scripts.interp.backfill_running_stats
        to retrofit).
        """
        self._require_open()
        gpath = group_path(name, tags or {})
        stats_path = f"{gpath}/running_stats"
        source = self._stats_file if self._stats_file is not None else self._files[0]
        if stats_path not in source:
            return None
        grp = source[stats_path]
        return RunningStats(
            count=int(grp["count"][()]),
            mean=torch.from_numpy(np.asarray(grp["mean"][...])),
            M2=torch.from_numpy(np.asarray(grp["M2"][...])),
        )

    def read(
        self, name: str, tags: dict[str, str] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | np.ndarray]]:
        self._require_open()
        gpath = group_path(name, tags or {})
        act_path = f"{gpath}/activation"
        labels_path = f"{gpath}/labels"

        # Walk shards in order; a shard may legitimately omit a (name, tags)
        # combination if a gate dropped that target for that batch range, so
        # we skip absent shards silently rather than KeyError per-shard.
        parts: list[np.ndarray] = []
        label_parts: dict[str, list[np.ndarray]] = {}
        for f in self._files:
            if act_path not in f:
                continue
            parts.append(f[act_path][...])
            if labels_path in f:
                for label_name in f[labels_path].keys():
                    label_parts.setdefault(label_name, []).append(
                        f[f"{labels_path}/{label_name}"][...]
                    )
        if not parts:
            raise KeyError(f"no activation at {act_path}")
        activation = torch.from_numpy(np.concatenate(parts, axis=0))
        # String labels are stored as vlen utf-8 (numpy object dtype on
        # readback), which torch.from_numpy refuses. Mirror the writer side
        # of the contract — ActivationRecord.per_cell already allows
        # `Tensor | ndarray` — and return numpy for object dtypes.
        labels: dict[str, torch.Tensor | np.ndarray] = {}
        for k, v in label_parts.items():
            joined = np.concatenate(v, axis=0)
            if joined.dtype == object:
                # h5py vlen utf-8 round-trips as bytes in object arrays;
                # decode so callers get str-valued ndarrays (matches what
                # the writer received). torch has no string dtype so we
                # stay on numpy for this branch.
                labels[k] = np.array(
                    [x.decode() if isinstance(x, bytes) else x for x in joined],
                    dtype=object,
                )
            else:
                labels[k] = torch.from_numpy(joined)
        return activation, labels

    def _require_open(self) -> None:
        if not self._files:
            raise RuntimeError("H5ActivationReader not open (use as context manager)")


def _attrs_equal(a: Any, b: Any) -> bool:
    # h5py returns numpy scalars / arrays for attrs; element-wise compare so
    # ["a", "b"] equals np.array(["a", "b"]). Bytes/str mismatch is common
    # across file handles; normalize both.
    if isinstance(a, bytes):
        a = a.decode()
    if isinstance(b, bytes):
        b = b.decode()
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if a_arr.shape != b_arr.shape:
            return False
        return bool(np.array_equal(a_arr, b_arr))
    return a == b
