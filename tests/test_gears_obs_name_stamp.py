from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.data.gears import _stamp_obs_names


class _StubData:
    """Stand-in for torch_geometric.data.Data; we only need attribute access."""

    def __init__(self, **kw: object) -> None:
        for k, v in kw.items():
            setattr(self, k, v)


def _build_pkl(
    tmp_path: Path, conds: dict[str, int]
) -> tuple[Path, SimpleNamespace]:
    """Write a synthetic cell_graphs.pkl and a matching pert_data stub.

    `conds` maps condition_str -> n_cells. The stub's adata has obs.condition
    laid out in that order with deterministic obs_names like "<cond>_cell_<i>".
    """
    pkl_path = tmp_path / "cell_graphs.pkl"
    cell_graphs = {
        cond: [_StubData() for _ in range(n)] for cond, n in conds.items()
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(cell_graphs, f)

    obs_rows: list[tuple[str, str]] = []
    for cond, n in conds.items():
        for i in range(n):
            obs_rows.append((f"{cond}_cell_{i}", cond))
    obs_names = [r[0] for r in obs_rows]
    obs = pd.DataFrame(
        {"condition": [r[1] for r in obs_rows]},
        index=pd.Index(obs_names, name="cell"),
    )

    class _AdataView:
        # __getitem__ on the parent returns a view that exposes obs_names.
        def __init__(self, names: np.ndarray) -> None:
            self._names = names

        @property
        def obs_names(self) -> "_ObsNames":
            return _ObsNames(self._names)

    class _ObsNames:
        def __init__(self, names: np.ndarray) -> None:
            self._names = names

        def to_numpy(self) -> np.ndarray:
            return self._names

    class _Adata:
        # Indexing with a boolean Series returns a slice with obs_names.
        def __init__(self, obs_df: pd.DataFrame) -> None:
            self.obs = obs_df

        def __getitem__(self, mask: pd.Series) -> _AdataView:
            return _AdataView(np.asarray(self.obs.index[mask]))

    pert_data = SimpleNamespace(adata=_Adata(obs))
    return pkl_path, pert_data


def test_stamp_writes_obs_name_per_data(tmp_path: Path) -> None:
    pkl_path, pert_data = _build_pkl(
        tmp_path, {"A+ctrl": 3, "B+ctrl": 2, "ctrl": 4}
    )

    _stamp_obs_names(pert_data, pkl_path)

    with open(pkl_path, "rb") as f:
        cell_graphs = pickle.load(f)
    # Per-condition obs_names must match the synthetic naming, in order.
    assert [d.obs_name for d in cell_graphs["A+ctrl"]] == [
        "A+ctrl_cell_0", "A+ctrl_cell_1", "A+ctrl_cell_2",
    ]
    assert [d.obs_name for d in cell_graphs["B+ctrl"]] == [
        "B+ctrl_cell_0", "B+ctrl_cell_1",
    ]
    assert [d.obs_name for d in cell_graphs["ctrl"]] == [
        "ctrl_cell_0", "ctrl_cell_1", "ctrl_cell_2", "ctrl_cell_3",
    ]


def test_stamp_is_semantically_idempotent(tmp_path: Path) -> None:
    pkl_path, pert_data = _build_pkl(tmp_path, {"A+ctrl": 2})
    _stamp_obs_names(pert_data, pkl_path)
    _stamp_obs_names(pert_data, pkl_path)  # second pass must not corrupt

    with open(pkl_path, "rb") as f:
        cell_graphs = pickle.load(f)
    assert [d.obs_name for d in cell_graphs["A+ctrl"]] == [
        "A+ctrl_cell_0", "A+ctrl_cell_1",
    ]


def test_stamp_raises_on_length_mismatch(tmp_path: Path) -> None:
    # Pkl has 3 graphs for "A+ctrl"; adata has only 2 cells under that
    # condition. Indicates num_samples != 1 or a corrupted pkl — refuse.
    pkl_path, pert_data = _build_pkl(tmp_path, {"A+ctrl": 2})
    with open(pkl_path, "rb") as f:
        cell_graphs = pickle.load(f)
    cell_graphs["A+ctrl"].append(_StubData())  # 3 graphs, 2 obs rows
    with open(pkl_path, "wb") as f:
        pickle.dump(cell_graphs, f)

    with pytest.raises(RuntimeError, match="length mismatch"):
        _stamp_obs_names(pert_data, pkl_path)
