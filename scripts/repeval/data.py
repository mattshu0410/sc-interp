"""CaptureView: shared loader for repeval scripts.

Wraps a capture folder (`shard-*.h5` + `stats.h5`). Layer arg is the
full capture name as written by HookManager (e.g.
`"transformer_encoder.layers.7"`); validated against what's in the shard.
"""
from __future__ import annotations

from functools import cached_property
from pathlib import Path

import h5py
import numpy as np

from scripts.repeval.conditions import resolve


def _decode(arr: np.ndarray) -> np.ndarray:
    """h5 returns object arrays of bytes; decode to str array."""
    return np.array(
        [c.decode() if isinstance(c, (bytes, bytearray)) else str(c) for c in arr]
    )


def _discover_groups(shard: Path, phase: str) -> list[str]:
    """Capture names with an `activation` dataset under the given phase tag."""
    found: list[str] = []
    phase_seg = f"phase={phase}"

    def visit(name: str, obj: object) -> None:
        if not (isinstance(obj, h5py.Dataset) and name.endswith("/activation")):
            return
        parts = name.split("/")
        if len(parts) < 3 or parts[-2] != phase_seg:
            return
        found.append("/".join(parts[:-2]))

    with h5py.File(shard) as f:
        f.visititems(visit)
    return found


class CaptureView:
    """Read activations + labels from a capture folder.

    Args:
        condition: registered name (``"base"``/``"esm"``/``"random"``) or
            a Path to a capture folder.
        layer: full capture name as written by HookManager
            (e.g. ``"transformer_encoder.layers.7"``).
        phase: capture-time phase tag (default ``"predict"``).
    """

    def __init__(
        self,
        condition: str | Path,
        layer: str,
        *,
        phase: str = "predict",
    ):
        self.path = resolve(condition)
        self.phase = phase
        self._shards = sorted(self.path.glob("shard-*.h5"))
        if not self._shards:
            raise FileNotFoundError(f"no shard-*.h5 in {self.path}")
        self._stats_path = self.path / "stats.h5"
        if not self._stats_path.exists():
            raise FileNotFoundError(f"missing stats.h5 in {self.path}")
        groups = _discover_groups(self._shards[0], phase=phase)
        if layer not in groups:
            raise ValueError(
                f"layer {layer!r} not in {self._shards[0].name} under phase={phase}; "
                f"available: {sorted(groups)}"
            )
        self.layer = layer
        self._gpath = f"{layer}/phase={phase}"

    # ── Metadata ─────────────────────────────────────────────────────────
    @cached_property
    def stats(self) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, std)`` per feature, shape ``(d,)`` each. Reads sidecar once."""
        with h5py.File(self._stats_path) as f:
            g = f[f"{self._gpath}/running_stats"]
            mean = g["mean"][:].astype(np.float64)
            m2 = g["M2"][:].astype(np.float64)
            count = float(g["count"][()])
        std = np.sqrt(m2 / count)
        # Guard near-zero dims so normalize() doesn't divide by ~0.
        std = np.where(std > 1e-6, std, 1.0)
        return mean, std

    @cached_property
    def n_cells(self) -> int:
        n = 0
        for sh in self._shards:
            with h5py.File(sh) as f:
                n += f[f"{self._gpath}/activation"].shape[0]
        return n

    @cached_property
    def feature_dim(self) -> int:
        with h5py.File(self._shards[0]) as f:
            return int(f[f"{self._gpath}/activation"].shape[-1])

    @cached_property
    def tokens_per_cell(self) -> int:
        with h5py.File(self._shards[0]) as f:
            shape = f[f"{self._gpath}/activation"].shape
        # BTD layout: (n, T, d). If 2-D (BD), tokens_per_cell == 1.
        return int(shape[1]) if len(shape) == 3 else 1

    # ── Activations ──────────────────────────────────────────────────────
    def per_token(
        self, n_cells: int | None = None, *, normalize: bool = True
    ) -> np.ndarray:
        """First ``n_cells`` activations as ``(n, T, d)``.

        If ``normalize``, applies per-feature z-score using the sidecar stats.
        Walks shards in order and stops once ``n_cells`` is satisfied.
        """
        n_target = self.n_cells if n_cells is None else min(n_cells, self.n_cells)
        chunks: list[np.ndarray] = []
        seen = 0
        for sh in self._shards:
            with h5py.File(sh) as f:
                ds = f[f"{self._gpath}/activation"]
                take = min(ds.shape[0], n_target - seen)
                if take <= 0:
                    break
                chunks.append(ds[:take].astype(np.float32))
                seen += take
                if seen >= n_target:
                    break
        x = chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)
        if normalize:
            mean, std = self.stats
            x = (x - mean.astype(np.float32)) / std.astype(np.float32)
        return x

    def per_cell(
        self,
        n_cells: int | None = None,
        *,
        normalize: bool = True,
        pool: str = "mean",
        chunk_size: int = 1024,
    ) -> np.ndarray:
        """Pool tokens to one vector per cell. Returns ``(n, d)``.

        Streams shards in ``chunk_size``-cell blocks, normalizing then
        pooling each block. Peak memory is ``chunk_size × T × d``.

        Args:
            pool: ``"mean"`` (default) or ``"max"``. ``"max"`` takes
                element-wise max across token positions, preserving
                signal concentrated in a small number of tokens.
        """
        if pool not in ("mean", "max"):
            raise ValueError(f"pool must be 'mean' or 'max', got {pool!r}")
        n_target = self.n_cells if n_cells is None else min(n_cells, self.n_cells)
        if normalize:
            mean, std = self.stats
            mean32 = mean.astype(np.float32)
            std32 = std.astype(np.float32)
        pooled: list[np.ndarray] = []
        seen = 0
        for sh in self._shards:
            if seen >= n_target:
                break
            with h5py.File(sh) as f:
                ds = f[f"{self._gpath}/activation"]
                n_in_shard = ds.shape[0]
                for start in range(0, n_in_shard, chunk_size):
                    if seen >= n_target:
                        break
                    stop = min(start + chunk_size, n_in_shard, start + (n_target - seen))
                    chunk = ds[start:stop].astype(np.float32)
                    if normalize:
                        chunk = (chunk - mean32) / std32
                    if chunk.ndim == 3:
                        chunk = chunk.max(axis=1) if pool == "max" else chunk.mean(axis=1)
                    pooled.append(chunk)
                    seen += chunk.shape[0]
        return pooled[0] if len(pooled) == 1 else np.concatenate(pooled, axis=0)

    # ── Labels ───────────────────────────────────────────────────────────
    def labels(self, n_cells: int | None = None) -> dict[str, np.ndarray]:
        """Per-cell labels: ``cell_id``, ``pert``, ``cell_line``.

        ``cell_line`` is parsed from the ``cell_id`` suffix
        (e.g. ``CTGTGAATCCCTCATG-13-rpe1`` → ``rpe1``).
        """
        n_target = self.n_cells if n_cells is None else min(n_cells, self.n_cells)
        cell_ids: list[np.ndarray] = []
        perts: list[np.ndarray] = []
        seen = 0
        for sh in self._shards:
            with h5py.File(sh) as f:
                ldir = f[f"{self._gpath}/labels"]
                avail = ldir["cell_id"].shape[0]
                take = min(avail, n_target - seen)
                if take <= 0:
                    break
                cell_ids.append(ldir["cell_id"][:take])
                perts.append(ldir["pert"][:take])
                seen += take
                if seen >= n_target:
                    break
        cid = _decode(np.concatenate(cell_ids))
        pert = _decode(np.concatenate(perts))
        cell_line = np.array([c.split("-")[-1] for c in cid])
        return {"cell_id": cid, "pert": pert, "cell_line": cell_line}

    # ── Repr ─────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return f"CaptureView({self.path.name}, {self.layer}, n={self.n_cells})"
