from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from scripts.interp.hook_sinks import H5ActivationSink, group_path
from scripts.interp.hooks import Layout  # re-exported for callers


class H5ActivationReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._file: h5py.File | None = None

    def __enter__(self) -> H5ActivationReader:
        self._file = h5py.File(self.path, "r")
        version = self._file["meta"].attrs.get("schema_version")
        if version != H5ActivationSink.SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {version!r} at {self.path} does not match "
                f"reader expected {H5ActivationSink.SCHEMA_VERSION!r}"
            )
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def meta(self) -> dict[str, Any]:
        self._require_open()
        assert self._file is not None
        out: dict[str, Any] = {}
        for k, v in self._file["meta"].attrs.items():
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
        assert self._file is not None
        gpath = group_path(name, tags or {})
        act_path = f"{gpath}/activation"
        if act_path not in self._file:
            raise KeyError(f"no activation at {act_path}")
        attr = self._file[act_path].attrs.get("layout", "")
        if isinstance(attr, bytes):
            attr = attr.decode()
        return str(attr)  # type: ignore[return-value]

    def read(
        self, name: str, tags: dict[str, str] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._require_open()
        assert self._file is not None
        gpath = group_path(name, tags or {})
        act_path = f"{gpath}/activation"
        if act_path not in self._file:
            raise KeyError(f"no activation at {act_path}")
        activation = torch.from_numpy(self._file[act_path][...])
        labels: dict[str, torch.Tensor] = {}
        labels_path = f"{gpath}/labels"
        if labels_path in self._file:
            labels_grp = self._file[labels_path]
            for label_name in labels_grp.keys():
                labels[label_name] = torch.from_numpy(labels_grp[label_name][...])
        return activation, labels

    def _require_open(self) -> None:
        if self._file is None:
            raise RuntimeError("H5ActivationReader not open (use as context manager)")
