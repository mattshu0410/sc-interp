"""Correlate per-feature activation with per-perturbation ESM-vs-base gain.

Joins a feature's ``by_pert`` marginals with eval-derived
``pearson_delta`` gains. Pearson and Spearman r quantify the extent to
which a feature's activation tracks ESM's outperformance — correlational
evidence that the feature is involved in driving the gap.

This is *not* causal evidence. A high correlation could reflect a
feature that drives the gain, a downstream effect of the cause, or a
confound (feature fires on ribosomal genes, ESM also wins on
ribosome-biogenesis perts, but for unrelated reasons). Causal validation
requires ablation (deferred to a follow-up).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats as scistats


def gain_correlation(
    card: dict,
    gains: pd.DataFrame,
    *,
    activation_field: str = "mean",
    gain_field: str = "gain",
    min_perts: int = 10,
    min_active_perts: int = 3,
) -> dict:
    """Pearson + Spearman correlation between per-pert activation and gain.

    Args:
        card: feature card dict (from :class:`FeatureCard.to_json_dict`).
        gains: index=perturbation, columns include ``gain_field``.
        activation_field: which `by_pert[p]` stat to use (default mean f).
        gain_field: which column of `gains` to correlate against.
        min_perts: minimum perts overlapping between card and gains
            DataFrame for correlation to be meaningful (else returns NaN).
        min_active_perts: minimum perts with non-zero activation; if
            fewer the feature is silent on most perts and the
            correlation is dominated by zeros (returns NaN).

    Returns a dict with both correlations, the perturbation count, and
    the top-5 perts by ``activation × gain`` (the perts that contribute
    most to a positive correlation, useful as paper-table examples).
    """
    by_pert = card.get("by_pert", {})
    rows = []
    for pert, stats in by_pert.items():
        if pert not in gains.index:
            continue
        rows.append((
            pert,
            float(stats.get(activation_field, 0.0)),
            float(gains.loc[pert, gain_field]),
        ))
    if len(rows) < min_perts:
        return _empty(reason=f"only {len(rows)} overlapping perts (< {min_perts})")

    df = pd.DataFrame(rows, columns=["pert", "activation", "gain"])
    n_active = int((df["activation"] > 0).sum())
    if n_active < min_active_perts:
        return _empty(reason=f"only {n_active} active perts (< {min_active_perts})", n_perts=len(df))

    pearson_r, pearson_p = scistats.pearsonr(df["activation"], df["gain"])
    spearman_r, spearman_p = scistats.spearmanr(df["activation"], df["gain"])

    # Top perts by activation × gain (signed product). Positive products
    # contribute to a positive correlation.
    df["product"] = df["activation"] * df["gain"]
    top = df.nlargest(5, "product")[["pert", "activation", "gain", "product"]].to_dict("records")

    return {
        "n_perts": len(df),
        "n_active_perts": n_active,
        "pearson_r": float(pearson_r) if math.isfinite(pearson_r) else None,
        "pearson_p": float(pearson_p) if math.isfinite(pearson_p) else None,
        "spearman_r": float(spearman_r) if math.isfinite(spearman_r) else None,
        "spearman_p": float(spearman_p) if math.isfinite(spearman_p) else None,
        "top_perts_by_activation_x_gain": top,
    }


def _empty(*, reason: str, n_perts: int = 0) -> dict:
    return {
        "n_perts": n_perts,
        "n_active_perts": 0,
        "pearson_r": None,
        "pearson_p": None,
        "spearman_r": None,
        "spearman_p": None,
        "top_perts_by_activation_x_gain": [],
        "skipped_reason": reason,
    }
