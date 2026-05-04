from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from scripts.interp.hooks import ActivationRecord


@dataclass(frozen=True)
class RunningStats:
    """Welford accumulator state for one (capture, tags) group.

    M2 is sum of squared deviations from the mean. Mean and M2 are stored
    at whatever floating-point dtype the source activations carried — no
    silent promotion.
    """

    count: int
    mean: torch.Tensor
    M2: torch.Tensor

    def std(self, unbiased: bool = False) -> torch.Tensor:
        denom = self.count - (1 if unbiased else 0)
        if denom <= 0:
            return torch.zeros_like(self.M2)
        return (self.M2 / denom).sqrt()

    @classmethod
    def merge(cls, a: "RunningStats", b: "RunningStats") -> "RunningStats":
        # Chan's parallel Welford merge.
        if a.count == 0:
            return b
        if b.count == 0:
            return a
        n = a.count + b.count
        delta = b.mean - a.mean
        mean = a.mean + delta * (b.count / n)
        M2 = a.M2 + b.M2 + delta.pow(2) * (a.count * b.count / n)
        return cls(count=n, mean=mean, M2=M2)


class _RunningStatWelford:
    """Mutable per-write accumulator; sink snapshots it at flush time.

    Accumulator dtype matches the first batch's float dtype. Integer
    activations get promoted to float32 since fractional means need a
    floating-point representation.
    """

    __slots__ = ("count", "mean", "M2", "_feature_shape", "_dtype")

    def __init__(self) -> None:
        self.count = 0
        self.mean: torch.Tensor | None = None
        self.M2: torch.Tensor | None = None
        self._feature_shape: tuple[int, ...] | None = None
        self._dtype: torch.dtype | None = None

    def update(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        x = x.detach().cpu()
        if not x.dtype.is_floating_point:
            x = x.to(torch.float32)

        if self._dtype is None:
            self._dtype = x.dtype
        elif x.dtype != self._dtype:
            x = x.to(self._dtype)

        batch_size = x.shape[0]
        if self._feature_shape is None:
            self._feature_shape = tuple(x.shape[1:])
        elif tuple(x.shape[1:]) != self._feature_shape:
            raise ValueError(
                f"running_stats: feature shape changed mid-stream from "
                f"{self._feature_shape} to {tuple(x.shape[1:])}"
            )

        if batch_size == 1:
            # Within-batch variance is 0; avoid NaN from var() on length-1.
            batch_mean = x[0]
            batch_M2 = torch.zeros_like(batch_mean)
        else:
            batch_mean = x.mean(dim=0)
            batch_M2 = (x - batch_mean).pow(2).sum(dim=0)

        if self.count == 0:
            self.mean = batch_mean.clone()
            self.M2 = batch_M2.clone()
            self.count = batch_size
            return

        n = self.count + batch_size
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_size / n)
        self.M2 = self.M2 + batch_M2 + delta.pow(2) * (self.count * batch_size / n)
        self.count = n

    def snapshot(self) -> RunningStats | None:
        if self.count == 0 or self.mean is None or self.M2 is None:
            return None
        return RunningStats(
            count=int(self.count),
            mean=self.mean.clone(),
            M2=self.M2.clone(),
        )

    def reset(self) -> None:
        self.count = 0
        self.mean = None
        self.M2 = None
        self._feature_shape = None
        self._dtype = None


def default_activation_out(
    repo_root: Path, runner: str, dataset: str, split: str
) -> Path:
    # Canonical on-disk location for captured activations. Shared vocabulary
    # with downstream probe/SAE/patching scripts so they can find files by
    # (runner, dataset, split) without duplicating the f-string.
    return repo_root / "predictions" / f"{runner}_{dataset}_{split}.activations.h5"


def _escape_segment(s: str) -> str:
    # Percent-encode characters that carry structural meaning in the h5
    # tag-subpath schema: '/' (group separator), '=' (key/value delimiter
    # inside a tag segment), and '%' itself (so the encoding is reversible).
    # Without this, `{"layer": "1/2"}` or `name="blocks/0"` silently create
    # ambiguous nested groups that collide with legit names.
    return s.replace("%", "%25").replace("/", "%2F").replace("=", "%3D")


def _stats_view(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    # BTD: collapse the token axis into samples so running_stats are per-feature-dim.
    if layout == "BTD" and tensor.ndim == 3:
        return tensor.reshape(-1, tensor.shape[-1])
    return tensor


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
        # Callers count batches for assertions; kept even though this sink
        # doesn't shard so the sink protocol is uniform across impls.
        self.batch_end_count = 0

    def write(self, record: ActivationRecord) -> None:
        if self._closed:
            raise RuntimeError("write() called on closed MemoryActivationSink")
        self.records.append(record)

    def batch_end(self) -> None:
        # No-op: in-memory sink has nothing to rotate. The counter exists
        # so test spies can assert HookManager / extractors fire the hook
        # exactly once per run() without a dedicated spy sink class.
        if self._closed:
            raise RuntimeError("batch_end() called on closed MemoryActivationSink")
        self.batch_end_count += 1

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> MemoryActivationSink:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class H5ActivationSink:
    # v3: per-cell labels may be np.ndarray (incl. object/string), and
    # extra_meta values may be non-string (arrays, ints) — e.g., gene_symbols
    # or num_genes. Readers can branch on this to parse string labels.
    # `running_stats/{count, mean, M2}` is an optional, backwards-compatible
    # sidecar under each `<capture>/<tags>/` group. Welford accumulators
    # populated at write time so training-time normalisation can skip the
    # streaming pass. Older shards lacking the group fall through to a
    # streaming computation downstream — additive change, no version bump.
    SCHEMA_VERSION = "3"
    REQUIRED_META_FIELDS = ("runner", "dataset", "split", "capture_names")
    STATS_FILE_NAME = "stats.h5"

    # When a caller passes `batches_per_shard`, shards are written to a
    # folder with this name format. Zero-padding width chosen for
    # lexicographic sort == numeric sort up to 99,999 shards — more than
    # enough for realistic captures (100k shards × 1GB = 100TB).
    SHARD_NAME_FORMAT = "shard-{:05d}.h5"
    _SHARD_GLOB = "shard-*.h5"

    def __init__(
        self,
        path: str | Path,
        *,
        runner: str,
        dataset: str,
        split: str,
        capture_names: list[str],
        git_sha: str | None = None,
        extra_meta: dict[str, Any] | None = None,
        compression_opts: int = 4,
        mode: str = "x",
        batches_per_shard: int | None = None,
        compute_running_stats: bool = True,
    ) -> None:
        if not capture_names:
            raise ValueError("capture_names must be a non-empty list")
        if mode not in ("x", "w"):
            raise ValueError(
                f"mode must be 'x' (refuse overwrite, default) or 'w' "
                f"(truncate), got {mode!r}"
            )
        if batches_per_shard is not None and batches_per_shard <= 0:
            raise ValueError(
                f"batches_per_shard must be positive, got {batches_per_shard!r}"
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
        # When set, `path` is treated as a FOLDER and writes go to
        # shard-NNNNN.h5 files rotated every N `batch_end()` calls. None
        # preserves the single-file layout.
        self.batches_per_shard = batches_per_shard
        self.compute_running_stats = compute_running_stats
        # One global accumulator per (capture, tags) — kept across shard
        # rotations; flushed once at close. Folder mode → `<folder>/stats.h5`,
        # single-file mode → in the file under the activation's group.
        self._running_stats: dict[
            tuple[str, tuple[tuple[str, str], ...]], _RunningStatWelford
        ] = {}
        self._file: h5py.File | None = None
        self._closed = False
        self._shard_index = 0
        self._batches_in_shard = 0

    def __enter__(self) -> H5ActivationSink:
        if self.batches_per_shard is None:
            # Single-file mode — unchanged from v3.
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.mode == "x" and self.path.exists():
                raise FileExistsError(
                    f"{self.path} already exists; pass mode='w' to overwrite"
                )
            self._file = self._open_and_init(self.path)
        else:
            self._prepare_shard_folder()
            self._file = self._open_and_init(self._shard_path(self._shard_index))
        return self

    def _shard_path(self, index: int) -> Path:
        return self.path / self.SHARD_NAME_FORMAT.format(index)

    def _prepare_shard_folder(self) -> None:
        # 'x': must not exist. 'w': may exist; wipe existing shard files but
        # refuse to touch unrelated files so a mistyped path can't clobber
        # the user's data.
        if self.path.exists() and self.path.is_file():
            raise NotADirectoryError(
                f"{self.path} exists as a file; sharded mode requires a "
                "folder path (or a path that does not yet exist)"
            )
        if self.mode == "x" and self.path.exists():
            raise FileExistsError(
                f"{self.path} already exists; pass mode='w' to overwrite"
            )
        if self.mode == "w" and self.path.exists():
            children = list(self.path.iterdir())
            unrelated = [
                c for c in children
                if not (c.is_file() and c.name.startswith("shard-") and c.suffix == ".h5")
            ]
            if unrelated:
                names = ", ".join(sorted(c.name for c in unrelated))
                raise FileExistsError(
                    f"{self.path} contains non-shard files ({names}); refusing "
                    "to wipe. Remove them manually or choose a fresh path."
                )
            for shard in children:
                shard.unlink()
        self.path.mkdir(parents=True, exist_ok=True)

    def _open_and_init(self, file_path: Path) -> h5py.File:
        # Shared between single-file mode and every new shard — each shard
        # carries its own /meta so it's independently openable/verifiable.
        file_path.parent.mkdir(parents=True, exist_ok=True)
        f = h5py.File(file_path, "w")
        meta_grp = f.create_group("meta")
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
            # Array-typed values (e.g. norman's ~5k gene_symbols) overflow the
            # 64KB HDF5 object-header limit when stored as attrs. Route any
            # array/list to a dataset under /meta/<k>; scalars stay as attrs.
            if isinstance(v, (np.ndarray, list, tuple)):
                arr = np.asarray(v)
                if arr.dtype == object:
                    meta_grp.create_dataset(
                        k, data=arr, dtype=h5py.string_dtype(encoding="utf-8")
                    )
                else:
                    meta_grp.create_dataset(k, data=arr)
            else:
                meta_grp.attrs[k] = v
        return f

    def write(self, record: ActivationRecord) -> None:
        if self._closed:
            raise RuntimeError("write() called on closed H5ActivationSink")
        f = self._ensure_file_open()
        gpath = group_path(record.name, record.metadata_tags)
        arr = record.tensor.detach().cpu().numpy()
        act_path = f"{gpath}/activation"
        if act_path in f:
            dset = f[act_path]
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
            dset = f.create_dataset(
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

        for label_name, label_value in record.per_cell.items():
            # torch tensors: standard detach→cpu→numpy. np.ndarray: pass
            # through (typically strings as dtype=object, which torch can't
            # carry). This is the path that makes pert labels self-describing
            # in the h5.
            if isinstance(label_value, torch.Tensor):
                label_arr = label_value.detach().cpu().numpy()
            else:
                label_arr = np.asarray(label_value)
            label_path = f"{gpath}/labels/{label_name}"
            if label_path in f:
                ldset = f[label_path]
                if ldset.dtype != label_arr.dtype and label_arr.dtype != object:
                    # object-dtype inputs land in a vlen-string dataset whose
                    # h5py dtype reads back as object but compares != to the
                    # stored string_dtype, so skip the strict check for them.
                    raise ValueError(
                        f"H5 append dtype mismatch at {label_path}: existing "
                        f"{ldset.dtype} != new {label_arr.dtype}"
                    )
                old = ldset.shape[0]
                ldset.resize((old + label_arr.shape[0],) + label_arr.shape[1:])
                ldset[old:] = label_arr
            else:
                lmaxshape = (None,) + label_arr.shape[1:]
                create_kwargs: dict[str, Any] = {
                    "maxshape": lmaxshape,
                    "chunks": True,
                    "compression": "gzip",
                    "compression_opts": self.compression_opts,
                }
                if label_arr.dtype == object:
                    # Object-dtype numpy arrays are how we represent string
                    # labels (torch has no string dtype). h5py needs an
                    # explicit variable-length utf-8 dtype to serialize them.
                    create_kwargs["dtype"] = h5py.string_dtype(encoding="utf-8")
                f.create_dataset(
                    label_path, data=label_arr, **create_kwargs
                )

        if self.compute_running_stats:
            key = (record.name, tuple(sorted(record.metadata_tags.items())))
            acc = self._running_stats.setdefault(key, _RunningStatWelford())
            acc.update(_stats_view(record.tensor, record.layout))

    def _write_running_stats_to(self, f: h5py.File) -> None:
        # Idempotent dump of current accumulator snapshots into `f`.
        for (name, tags_tuple), acc in self._running_stats.items():
            snap = acc.snapshot()
            if snap is None:
                continue
            tags = dict(tags_tuple)
            stats_grp = f.require_group(f"{group_path(name, tags)}/running_stats")
            for ds_name, value in (
                ("count", np.int64(snap.count)),
                ("mean", snap.mean.numpy()),
                ("M2", snap.M2.numpy()),
            ):
                if ds_name in stats_grp:
                    del stats_grp[ds_name]
                stats_grp.create_dataset(ds_name, data=value)

    def _persist_running_stats(self) -> None:
        # Folder mode: refresh `<folder>/stats.h5` with the cumulative stats
        # so far. Single-file mode: write stats into the already-open file.
        # Mirrors upstream's save_state-after-each-shard pattern so a
        # mid-extraction crash leaves stats consistent with what's on disk.
        if not self.compute_running_stats or not self._running_stats:
            return
        if self.batches_per_shard is None:
            if self._file is None:
                return
            self._write_running_stats_to(self._file)
        else:
            stats_path = self.path / self.STATS_FILE_NAME
            with h5py.File(stats_path, "w") as f:
                self._write_running_stats_to(f)

    def batch_end(self) -> None:
        # Called by extractors after each run(). Single-file mode: no-op
        # (keeps the caller branch-free). Shard mode: count batches, rotate
        # when we hit the threshold. Rotation happens HERE, not at the next
        # write, so close() immediately after a threshold-hitting batch_end
        # does not leave a half-open empty shard.
        if self._closed:
            raise RuntimeError("batch_end() called on closed H5ActivationSink")
        if self.batches_per_shard is None:
            return
        self._batches_in_shard += 1
        if self._batches_in_shard >= self.batches_per_shard:
            # self._file may already be None here if this batch produced no
            # writes (e.g. gate dropped everything). Close only if open.
            if self._file is not None:
                self._file.close()
                self._file = None
            self._shard_index += 1
            self._batches_in_shard = 0
            self._persist_running_stats()

    def _ensure_file_open(self) -> h5py.File:
        # Hit by write() after a rotation — the next write lives in the next
        # shard, which we deferred opening in batch_end to avoid empty trailers.
        if self._file is None:
            if self._closed:
                raise RuntimeError("sink is closed")
            self._file = self._open_and_init(self._shard_path(self._shard_index))
        return self._file

    def close(self) -> None:
        if self._closed:
            return
        if self.batches_per_shard is None:
            # Single-file: stats live inside the file; flush before close.
            if self._file is not None:
                self._persist_running_stats()
                self._file.close()
                self._file = None
        else:
            # Folder: stats.h5 is independent of the active shard. Close
            # any open shard first, then write the final stats covering
            # batches since the last rotation.
            if self._file is not None:
                self._file.close()
                self._file = None
            self._persist_running_stats()
        self._closed = True

    def __exit__(self, *exc: object) -> None:
        self.close()
