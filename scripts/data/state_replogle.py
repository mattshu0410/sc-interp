"""State-Replogle-Filtered loader for cross-cell-line perturbation prediction.

Source for run_scgpt.py's `state_replogle` manifest type. Yields pyg Data
objects matching the shape contract that scGPT's pred_perturb consumes:

    Data.x: (n_genes, 2)  — col 0 = control basal, col 1 = pert one-hot flag
    Data.y: (1, n_genes)  — perturbed cell expression
    Data.pert: str        — perturbation label ('ctrl' for control)

Each perturbed cell is paired with one randomly sampled control cell from
the same cell line (matches GEARS' default num_samples=1).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from scripts.manifest import REPO_ROOT, Manifest


def _gears_condition(label: str, control_label: str) -> str:
    """Map a state pert label to GEARS condition format.

    State uses bare gene symbols ('TFAM') and 'non-targeting' for control;
    scGPT's compute_perturbation_metrics and rank_genes_groups_cov_all are
    keyed against the GEARS format ('TFAM+ctrl', 'ctrl').
    """
    return "ctrl" if label == control_label else f"{label}+ctrl"


def _compute_de(
    adata: ad.AnnData,
    train_lines: list[str],
    control_label: str,
    pert_col: str,
    cell_type_col: str,
) -> dict:
    """Build a rank_genes_groups_cov_all dict for the train cell lines.

    Filters to train_lines, sets the GEARS-required condition / cell_type
    obs columns, runs gears.data_utils.get_DE_genes (sc.tl.rank_genes_groups
    per cell_type with same-cell-line control as reference). Restricting
    to one cell type avoids ambiguity in scgpt's find_DE_genes which derives
    the cell-type prefix from the first dict key.
    """
    print(f"==> computing DE genes per perturbation on {train_lines}")
    from gears.data_utils import get_DE_genes

    de_adata = adata[adata.obs[cell_type_col].isin(train_lines)].copy()
    pert_arr = de_adata.obs[pert_col].astype(str).values
    de_adata.obs["condition"] = pd.Categorical(
        [_gears_condition(p, control_label) for p in pert_arr]
    )
    de_adata.obs["cell_type"] = de_adata.obs[cell_type_col].astype(str)
    get_DE_genes(de_adata, skip_calc_de=False)
    return de_adata.uns["rank_genes_groups_cov_all"]


def _read_X_subset(h5ad_path, keep_mask: np.ndarray) -> np.ndarray:
    """Read only the rows of /X selected by keep_mask, never materializing
    the full matrix in memory.

    The State-Filtered h5ad stores X as a contiguous dense float32 dataset
    (no chunking, no compression). h5py boolean indexing on a contiguous
    dataset reads only the selected rows from disk.
    """
    with h5py.File(h5ad_path, "r") as f:
        return f["X"][keep_mask, :]


@dataclass
class _Pair:
    perturbed_row: int   # index into the filtered adata's row axis
    basal_row: int       # same; used as control basal expression source
    pert_label: str      # gene symbol or 'ctrl'
    is_ctrl: bool


class _CellGraphDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        pairs: list[_Pair],
        gene_to_idx: dict[str, int],
        obs_names: np.ndarray,
    ) -> None:
        self.X = X
        self.pairs = pairs
        self.gene_to_idx = gene_to_idx
        self.obs_names = obs_names
        self.n_genes = X.shape[1]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Data:
        p = self.pairs[i]
        y_expr = self.X[p.perturbed_row]
        if p.is_ctrl:
            x_expr = y_expr
            flag = np.zeros(self.n_genes, dtype=np.float32)
            label = "ctrl"
        else:
            x_expr = self.X[p.basal_row]
            flag = np.zeros(self.n_genes, dtype=np.float32)
            gi = self.gene_to_idx.get(p.pert_label)
            if gi is not None:
                flag[gi] = 1.0
            # GEARS format so find_DE_genes can resolve rank_genes_groups_cov_all keys.
            label = f"{p.pert_label}+ctrl"
        x = torch.from_numpy(np.stack([x_expr.astype(np.float32), flag], axis=1))
        y = torch.from_numpy(y_expr.astype(np.float32)).unsqueeze(0)
        return Data(x=x, y=y, pert=label, obs_name=str(self.obs_names[p.perturbed_row]))


def _resolve_split(manifest: Manifest, args: argparse.Namespace) -> dict:
    """Resolve the named split to (train_cell_lines, test_cell_lines, val_pert_fraction)."""
    split_cfg = manifest.raw.get("split", {}) or {}
    name = getattr(args, "split_name", None) or split_cfg.get("default", "k562_to_rpe1")
    if name != "k562_to_rpe1":
        raise ValueError(f"unknown state_replogle split: {name!r}")
    return {
        "train_cell_lines": ["k562"],
        "test_cell_lines": ["rpe1"],
        "val_pert_fraction": float(split_cfg.get("val_pert_fraction", 0.05)),
    }


def _pairs_from_indices(
    indices: np.ndarray,
    pert_arr: np.ndarray,
    cell_line_arr: np.ndarray,
    *,
    control_label: str,
    ctrl_indices_per_line: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> list[_Pair]:
    """Emit one Pair per row in `indices`.

    Perturbed cells get a random basal from their cell line's full control
    pool. Control cells are emitted self-paired. Used by make_sample_loader;
    `_build_pairs` is the train/val/test analog that filters by perturbation
    set instead of taking explicit row indices.
    """
    pairs: list[_Pair] = []
    for row in indices:
        label = str(pert_arr[row])
        if label == control_label:
            pairs.append(_Pair(int(row), int(row), "ctrl", True))
        else:
            line = str(cell_line_arr[row])
            pool = ctrl_indices_per_line[line]
            basal = int(rng.choice(pool))
            pairs.append(_Pair(int(row), basal, label, False))
    return pairs


def make_sample_loader(
    adata: ad.AnnData,
    X: np.ndarray,
    indices: np.ndarray,
    *,
    pert_col: str,
    control_label: str,
    cell_type_col: str,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Build a pyg DataLoader over the rows in `indices`.

    Per-cell-line basal donors are drawn from the full control pool of each
    cell line (matches what scGPT saw at training time). Yields the same
    Data shape contract as the train/val/test loaders.
    """
    pert_arr = adata.obs[pert_col].values.astype(str)
    cell_line_arr = adata.obs[cell_type_col].values.astype(str)
    ctrl_mask = pert_arr == control_label
    ctrl_indices_per_line = {
        line: np.where(ctrl_mask & (cell_line_arr == line))[0]
        for line in np.unique(cell_line_arr)
    }
    for line, pool in ctrl_indices_per_line.items():
        if len(pool) == 0:
            raise RuntimeError(
                f"cell line {line!r} has no control cells; cannot pair basals"
            )

    pair_rng = np.random.default_rng(seed)
    pairs = _pairs_from_indices(
        indices, pert_arr, cell_line_arr,
        control_label=control_label,
        ctrl_indices_per_line=ctrl_indices_per_line,
        rng=pair_rng,
    )
    print(f"==> sample pairs: {len(pairs):,}")

    gene_to_idx = {g: i for i, g in enumerate(adata.var.index.astype(str))}
    # obs.index encodes cell-barcode + gem-group + cell-line (e.g.
    # 'AAACCCAAGAATAGTC-3-hepg2'), so it's globally unique and the right
    # cell_id for HookManager's per-cell metadata.
    obs_names = adata.obs.index.values.astype(str)
    ds = _CellGraphDataset(X, pairs, gene_to_idx, obs_names)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def _build_pairs(
    pert_arr: np.ndarray,
    cell_line_mask: np.ndarray,
    perturbations: list[str],
    *,
    control_label: str,
    ctrl_indices: np.ndarray,
    rng: np.random.Generator,
) -> list[_Pair]:
    """Emit one Pair per cell whose perturbation is in `perturbations`.

    Perturbed cells get a random basal row from `ctrl_indices`. Control
    cells are emitted self-paired (basal == perturbed).
    """
    pert_set = set(perturbations)
    pairs: list[_Pair] = []
    for row in np.where(cell_line_mask)[0]:
        label = str(pert_arr[row])
        if label not in pert_set:
            continue
        if label == control_label:
            pairs.append(_Pair(int(row), int(row), "ctrl", True))
        else:
            basal = int(rng.choice(ctrl_indices))
            pairs.append(_Pair(int(row), basal, label, False))
    return pairs


def load_state_replogle(
    manifest: Manifest,
    args: argparse.Namespace,
) -> tuple[
    DataLoader, DataLoader, DataLoader, ad.AnnData, pd.DataFrame, DataLoader | None
]:
    """Build (train, val, test, ctrl_adata, var, sample) for state_replogle.

    Test loader restricted to the perturbed-gene overlap between train and
    test cell lines (isolates cross-cell-line transfer from unseen-pert
    generalization). `sample` is populated only when args.split == 'sample';
    it covers train + test cell lines balanced by args.sample_by.
    """
    pert_col = manifest.obs.pert_col
    control_label = manifest.obs.control_label
    cell_type_col = manifest.obs.cell_type_col
    if cell_type_col is None:
        raise ValueError("state_replogle manifest needs obs.cell_type_col")

    cfg = _resolve_split(manifest, args)
    train_lines = cfg["train_cell_lines"]
    test_lines = cfg["test_cell_lines"]
    val_frac = cfg["val_pert_fraction"]

    h5ad_path = REPO_ROOT / "data" / manifest.name / manifest.files["adata"]
    print(f"==> opening {h5ad_path} (backed read for obs/var only)")
    backed = ad.read_h5ad(h5ad_path, backed="r")
    keep_mask = backed.obs[cell_type_col].isin(train_lines + test_lines).values
    print(f"==> reading X for {int(keep_mask.sum()):,} of {backed.n_obs:,} cells")
    X = _read_X_subset(h5ad_path, keep_mask)
    obs = backed.obs[keep_mask].copy()
    var = backed.var.copy()
    backed.file.close()
    adata = ad.AnnData(X=X, obs=obs, var=var)
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    print(f"==> loaded {adata.n_obs:,} cells, X={X.nbytes / 1e9:.1f} GB")

    pert_arr = adata.obs[pert_col].values.astype(str)
    train_mask = adata.obs[cell_type_col].isin(train_lines).values
    test_mask = adata.obs[cell_type_col].isin(test_lines).values

    train_perts_full = np.unique(pert_arr[train_mask])
    test_perts_full = np.unique(pert_arr[test_mask])
    train_perturbed = sorted(set(train_perts_full) - {control_label})
    test_perturbed = sorted(set(test_perts_full) - {control_label})
    overlap = sorted(set(train_perturbed) & set(test_perturbed))
    print(
        f"==> train perturbations: {len(train_perturbed):,}  "
        f"test perturbations: {len(test_perturbed):,}  "
        f"overlap (test eval set): {len(overlap):,}"
    )

    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_perturbed)
    n_val = max(1, int(round(len(train_perturbed) * val_frac)))
    val_perts = train_perturbed[:n_val]
    train_only_perts = train_perturbed[n_val:]
    test_perts_eval = overlap

    train_ctrl_indices = np.where(train_mask & (pert_arr == control_label))[0]
    test_ctrl_indices = np.where(test_mask & (pert_arr == control_label))[0]
    if len(train_ctrl_indices) == 0 or len(test_ctrl_indices) == 0:
        raise RuntimeError("each cell line needs at least one control cell")

    pair_rng = np.random.default_rng(args.seed + 1)
    train_pairs = _build_pairs(
        pert_arr, train_mask, train_only_perts + [control_label],
        control_label=control_label, ctrl_indices=train_ctrl_indices, rng=pair_rng,
    )
    val_pairs = _build_pairs(
        pert_arr, train_mask, val_perts,
        control_label=control_label, ctrl_indices=train_ctrl_indices, rng=pair_rng,
    )
    test_pairs = _build_pairs(
        pert_arr, test_mask, test_perts_eval,
        control_label=control_label, ctrl_indices=test_ctrl_indices, rng=pair_rng,
    )
    print(
        f"==> pairs train: {len(train_pairs):,}  val: {len(val_pairs):,}  test: {len(test_pairs):,}"
    )

    gene_to_idx = {g: i for i, g in enumerate(adata.var.index.astype(str))}
    # See make_sample_loader for why obs.index is the cell_id here.
    obs_names = adata.obs.index.values.astype(str)
    train_ds = _CellGraphDataset(X, train_pairs, gene_to_idx, obs_names)
    val_ds = _CellGraphDataset(X, val_pairs, gene_to_idx, obs_names)
    test_ds = _CellGraphDataset(X, test_pairs, gene_to_idx, obs_names)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False)

    # Train cell line's controls are the basal reference for compute_perturbation_metrics.
    ctrl_adata = adata[train_mask & (pert_arr == control_label)].copy()

    # Inject DE per perturbation so compute_perturbation_metrics can resolve
    # pearson_de / pearson_de_delta. Computed fresh each run.
    ctrl_adata.uns["rank_genes_groups_cov_all"] = _compute_de(
        adata, train_lines, control_label, pert_col, cell_type_col,
    )

    # scGPT runner expects a `gene_name` column in var; State's adata uses
    # the index for symbols. Mirror so build_gene_ids works without
    # special-casing source==state_replogle.
    var = adata.var.copy()
    if "gene_name" not in var.columns:
        var["gene_name"] = var.index.astype(str)

    sample_loader = None
    if getattr(args, "split", None) == "sample":
        from scripts.data.sampling import balanced_sample

        by = [c.strip() for c in args.sample_by.split(",") if c.strip()]
        indices = balanced_sample(
            adata.obs, by=by, n_per_bucket=args.sample_n_per_bucket, seed=args.seed,
        )
        print(
            f"==> sample: {len(indices):,} cells across "
            f"{len(adata.obs.groupby(by, observed=True))} buckets "
            f"(by={by}, n_per_bucket={args.sample_n_per_bucket})"
        )
        sample_loader = make_sample_loader(
            adata, X, indices,
            pert_col=pert_col, control_label=control_label,
            cell_type_col=cell_type_col,
            batch_size=args.eval_batch_size, seed=args.seed + 2,
        )

    return train_loader, val_loader, test_loader, ctrl_adata, var, sample_loader
