"""Backfill `running_stats` sidecars for h5 captures written before the
sidecar was added.

Single-file mode: writes the `running_stats/` group inside the file.
Folder mode (sharded): writes a `<folder>/stats.h5` sibling.

Idempotent — re-running overwrites existing stats. Streams activations in
chunks so memory stays O(chunk_rows × feature_dim).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.interp.hook_sinks import (
    H5ActivationSink,
    _RunningStatWelford,
    group_path,
)


_DEFAULT_CHUNK_ROWS = 256


def discover_groups(f: h5py.File) -> list[tuple[str, dict[str, str], str]]:
    """Walk the file and return every (capture_name, tags, h5_group_path)
    that has an `activation` dataset under it. The capture/tags split mirrors
    `group_path` so backfilled stats land at the same key the writer would
    have used."""
    found: list[tuple[str, dict[str, str], str]] = []

    def visit(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        if not isinstance(obj, h5py.Dataset) or not name.endswith("/activation"):
            return
        # name is the path relative to the file root, no leading slash.
        parts = name.split("/")
        # Last segment is "activation"; everything before is /<capture>/<tag_kvs>.
        capture = parts[0]
        tag_segments = parts[1:-1]
        tags: dict[str, str] = {}
        for seg in tag_segments:
            if "=" not in seg:
                # Unrecognised intermediate segment; skip the group entirely.
                return
            k, v = seg.split("=", 1)
            tags[_unescape(k)] = _unescape(v)
        capture = _unescape(capture)
        gpath = "/" + name.rsplit("/", 1)[0]
        found.append((capture, tags, gpath))

    f.visititems(visit)
    return found


def _unescape(s: str) -> str:
    return s.replace("%2F", "/").replace("%3D", "=").replace("%25", "%")


def accumulate_activation(
    sources: list[h5py.File],
    gpath: str,
    chunk_rows: int,
) -> _RunningStatWelford | None:
    """Stream rows of `<gpath>/activation` across `sources` (one or many
    shards) through a single Welford accumulator and return it."""
    acc = _RunningStatWelford()
    any_seen = False
    for f in sources:
        act_path = f"{gpath}/activation"
        if act_path not in f:
            continue
        any_seen = True
        dset = f[act_path]
        layout = dset.attrs.get("layout", "")
        if isinstance(layout, bytes):
            layout = layout.decode()
        n = dset.shape[0]
        for start in range(0, n, chunk_rows):
            stop = min(start + chunk_rows, n)
            chunk = torch.from_numpy(dset[start:stop])
            if layout == "BTD" and chunk.ndim == 3:
                chunk = chunk.reshape(-1, chunk.shape[-1])
            acc.update(chunk)
    return acc if any_seen else None


def write_into(file_or_folder: h5py.File, snapshots: list[tuple[str, dict[str, str], _RunningStatWelford]]) -> None:
    """Write each accumulator's snapshot into `file_or_folder` at
    `<group_path(capture, tags)>/running_stats/`."""
    for capture, tags, acc in snapshots:
        snap = acc.snapshot()
        if snap is None:
            continue
        stats_grp = file_or_folder.require_group(f"{group_path(capture, tags)}/running_stats")
        for ds_name, value in (
            ("count", np.int64(snap.count)),
            ("mean", snap.mean.numpy()),
            ("M2", snap.M2.numpy()),
        ):
            if ds_name in stats_grp:
                del stats_grp[ds_name]
            stats_grp.create_dataset(ds_name, data=value)


def backfill(
    path: Path,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
    captures: list[str] | None = None,
) -> int:
    """Backfill running_stats for an h5 file or shard folder.

    `captures` filters to specific capture names (e.g., ['transformer_encoder.layers.11']);
    None backfills every capture present.

    Returns the number of (capture, tags) groups that were stat-ed."""
    if path.is_dir():
        return _backfill_folder(path, chunk_rows, captures)
    return _backfill_single_file(path, chunk_rows, captures)


def _filter_groups(
    groups: list[tuple[str, dict[str, str], str]],
    captures: list[str] | None,
) -> list[tuple[str, dict[str, str], str]]:
    if captures is None:
        return groups
    selected = set(captures)
    return [g for g in groups if g[0] in selected]


def _backfill_single_file(
    path: Path, chunk_rows: int, captures: list[str] | None,
) -> int:
    with h5py.File(path, "r") as src:
        groups = _filter_groups(discover_groups(src), captures)
        accumulated: list[tuple[str, dict[str, str], _RunningStatWelford]] = []
        for capture, tags, gpath in groups:
            acc = accumulate_activation([src], gpath, chunk_rows)
            if acc is not None:
                accumulated.append((capture, tags, acc))
    with h5py.File(path, "a") as dst:
        write_into(dst, accumulated)
    return len(accumulated)


def _backfill_folder(
    folder: Path, chunk_rows: int, captures: list[str] | None,
) -> int:
    shard_paths = sorted(folder.glob(H5ActivationSink._SHARD_GLOB))
    if not shard_paths:
        raise FileNotFoundError(
            f"{folder} contains no {H5ActivationSink._SHARD_GLOB} files"
        )
    files = [h5py.File(p, "r") for p in shard_paths]
    try:
        # Discovery from the first shard is sufficient — meta consistency is
        # validated by the reader, so all shards carry the same group set.
        groups = _filter_groups(discover_groups(files[0]), captures)
        accumulated: list[tuple[str, dict[str, str], _RunningStatWelford]] = []
        for capture, tags, gpath in groups:
            acc = accumulate_activation(files, gpath, chunk_rows)
            if acc is not None:
                accumulated.append((capture, tags, acc))
    finally:
        for f in files:
            f.close()

    stats_path = folder / H5ActivationSink.STATS_FILE_NAME
    # Open in append mode so a partial backfill (specific captures) doesn't
    # wipe existing stats from earlier targeted runs. write_into deletes
    # the per-(capture,tags) running_stats group before re-creating, so
    # rerunning on the same captures is idempotent.
    with h5py.File(stats_path, "a") as dst:
        write_into(dst, accumulated)
    return len(accumulated)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="python -m scripts.interp.backfill_running_stats",
        description="Backfill running_stats sidecars on existing h5 captures.",
    )
    p.add_argument("path", type=Path, help="h5 file or shard folder.")
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=_DEFAULT_CHUNK_ROWS,
        help="Rows per Welford update.",
    )
    p.add_argument(
        "--captures",
        nargs="+",
        default=None,
        help="Restrict to specific capture names.",
    )
    args = p.parse_args(argv)
    n = backfill(args.path, chunk_rows=args.chunk_rows, captures=args.captures)
    dest = (
        args.path / H5ActivationSink.STATS_FILE_NAME
        if args.path.is_dir()
        else args.path
    )
    print(f"==> backfilled {n} (capture, tags) groups → {dest}")


if __name__ == "__main__":
    main()
