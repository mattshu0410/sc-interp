"""Source-neutral row-index samplers for activation collection.

Currently exports `balanced_sample`, which returns row indices balanced
across unique combinations of obs columns. Future selectors (e.g.
gap-slice for patching analysis) live here as sibling functions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def balanced_sample(
    obs: pd.DataFrame,
    by: list[str],
    n_per_bucket: int,
    seed: int,
) -> np.ndarray:
    """Greedy: up to n_per_bucket rows per unique combination of `by` columns.

    Buckets with fewer than n_per_bucket rows return all of them. Returned
    indices are sorted for deterministic h5 write order.
    """
    if not by:
        raise ValueError("balanced_sample requires at least one column in `by`")
    missing = [c for c in by if c not in obs.columns]
    if missing:
        raise KeyError(f"obs missing columns: {missing}")

    rng = np.random.default_rng(seed)
    indices: list[np.ndarray] = []
    for _, group_indices in obs.groupby(by, observed=True).indices.items():
        n = min(len(group_indices), n_per_bucket)
        sampled = rng.choice(group_indices, size=n, replace=False)
        indices.append(sampled)
    return np.sort(np.concatenate(indices))
