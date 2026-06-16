"""Generic per-perturbation eval-metric joiner.

Loads one metric column from N labelled eval-results CSVs and joins them
on the perturbation key, with optional gain columns ``gain_vs_<label>``
for each non-primary condition. No assumptions about runner, dataset,
or label names — caller decides everything.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_eval_metric(
    csv: Path,
    *,
    metric: str = "pearson_delta",
    pert_col: str = "perturbation",
) -> pd.Series:
    """Read one column from one eval CSV, indexed by perturbation."""
    df = pd.read_csv(csv)
    if pert_col not in df.columns:
        raise KeyError(f"{csv}: missing perturbation column {pert_col!r}; "
                       f"available: {list(df.columns)[:8]}...")
    if metric not in df.columns:
        raise KeyError(f"{csv}: missing metric column {metric!r}")
    return df.set_index(pert_col)[metric]


def join_eval_gains(
    primary_csv: Path,
    *,
    primary_label: str = "primary",
    baseline_csvs: dict[str, Path] | None = None,
    metric: str = "pearson_delta",
    pert_col: str = "perturbation",
) -> pd.DataFrame:
    """Inner-join one primary + N baseline eval CSVs on `pert_col`.

    Returns a DataFrame indexed by perturbation with one column per
    condition plus one ``gain_vs_<label>`` column per baseline (primary
    minus baseline).

    Caller decides what's "primary" and what's "baseline". For an
    ESM-vs-base study, pass primary=ESM and baseline_csvs={"base": ...}.
    To also condition against a random-prior model, add
    ``"random": ...`` to baseline_csvs.
    """
    cols = {primary_label: load_eval_metric(primary_csv, metric=metric, pert_col=pert_col)}
    for label, path in (baseline_csvs or {}).items():
        if label == primary_label:
            raise ValueError(f"baseline label {label!r} collides with primary_label")
        cols[label] = load_eval_metric(path, metric=metric, pert_col=pert_col)

    df = pd.DataFrame(cols).dropna(how="any")
    for label in (baseline_csvs or {}).keys():
        df[f"gain_vs_{label}"] = df[primary_label] - df[label]
    return df
