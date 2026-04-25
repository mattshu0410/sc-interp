"""GEARS-sourced dataset materialiser.

Downloads a GEARS-supported dataset, runs prepare_split to compute the
train/val/test partition, and writes the partition as a canonical split
JSON at the path scripts.data.splits expects.

The `gears` import happens inside materialise() so this module is safe
to import from venvs that lack gears.

Usage:
    source tools/.venv/bin/activate
    python -m scripts.data.gears --dataset norman
    python -m scripts.data.gears --dataset norman --split-type simulation --seed 42
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from scripts.data.splits import write_split
from scripts.manifest import REPO_ROOT, Manifest


def _stamp_obs_names(pert_data, pkl_path: Path) -> None:
    """Write per-cell `obs_name` (AnnData barcode) onto every Data in cell_graphs.pkl.

    GEARS' create_cell_graph_dataset iterates `for cell_z in adata_.X` where
    `adata_ = adata[obs.condition == cond]`, so under num_samples=1 (the gears
    default) the i-th Data in the per-condition list maps 1:1 to obs_names[i]
    of that slice. Without this stamp the capture h5 only carries cell_id
    (running index) and pert, leaving no way to join activations back to a
    specific source cell in the original AnnData.

    Idempotent: running it twice writes the same obs_name attr.
    """
    print(f"==> stamping obs_name onto Data objects in {pkl_path}")
    with open(pkl_path, "rb") as f:
        cell_graphs = pickle.load(f)

    adata = pert_data.adata
    obs_by_cond = adata.obs["condition"]
    stamped = 0
    for cond, data_list in cell_graphs.items():
        obs_names = adata[obs_by_cond == cond].obs_names.to_numpy()
        if len(data_list) != len(obs_names):
            # num_samples > 1 would produce len(data_list) == num_samples *
            # len(obs_names); the runner doesn't override num_samples today,
            # but if that changes, a stride mapping (i // num_samples) is
            # the fix. Refuse rather than silently misalign barcodes.
            raise RuntimeError(
                f"length mismatch for condition {cond!r}: "
                f"{len(data_list)} graphs vs {len(obs_names)} adata rows. "
                f"Likely num_samples != 1 in create_cell_graph_dataset."
            )
        for i, data in enumerate(data_list):
            data.obs_name = str(obs_names[i])
        stamped += len(data_list)

    with open(pkl_path, "wb") as f:
        pickle.dump(cell_graphs, f)
    print(f"    stamped {stamped} Data objects across {len(cell_graphs)} conditions")


def materialise(
    manifest: Manifest,
    split_type: str = "simulation",
    seed: int = 42,
    train_gene_set_size: float = 0.75,
) -> Path:
    """Run gears.PertData.load + prepare_split and emit a canonical split JSON.

    Returns the path of the JSON written. Raises ValueError if manifest.source
    is not 'gears'.
    """
    import numpy as np

    from gears import PertData

    if manifest.source != "gears":
        raise ValueError(
            f"gears materialiser requires manifest.source == 'gears', "
            f"got {manifest.source!r}"
        )

    gears_name = manifest.raw.get("gears_name")
    if not gears_name:
        raise ValueError(
            f"manifest for {manifest.name!r} missing raw['gears_name'] "
            f"required by the gears materialiser"
        )

    data_dir = REPO_ROOT / "data"
    print(f"==> materialising gears split for {manifest.name} ({gears_name})")
    print(f"    split_type={split_type}, seed={seed}, tgss={train_gene_set_size}")

    # PertData.load -> create_cell_graph_dataset draws control basals with
    # np.random.randint and never seeds it itself, so a fresh-clone rerun
    # produces a differently-paired cell_graphs.pkl. Seed the global numpy
    # RNG before load() so the .pkl (and therefore every downstream batch
    # order and shard layout) is reproducible from clean state.
    np.random.seed(seed)
    # default_pert_graph=False: matches scripts/run_gears.py. PertData caches
    # pert_idx in cell_graphs.pkl indexing whichever pert_names was active at
    # build time, and reuses the cache blindly on reload — if this flag
    # disagrees with the runner, the runner's model.forward does out-of-bounds
    # lookups into pert_emb (num_perts ≠ pkl's index space).
    pert_data = PertData(str(data_dir), default_pert_graph=False)
    pert_data.load(data_name=gears_name)

    # Add obs_name (AnnData cell barcode) to every Data in cell_graphs.pkl so
    # capture h5s can join back to the source AnnData by row, not just by
    # perturbation. Stock gears never writes this attr; we patch it post-load
    # so the cache stays consumable by both runner venvs (gears submodule and
    # scgpt's PyPI cell-gears) without a vendored library edit.
    _stamp_obs_names(
        pert_data, Path(data_dir) / gears_name / "data_pyg" / "cell_graphs.pkl"
    )

    pert_data.prepare_split(
        split=split_type,
        seed=seed,
        train_gene_set_size=train_gene_set_size,
    )

    set2conditions = pert_data.set2conditions
    out = write_split(
        dataset_name=manifest.name,
        split_type=split_type,
        seed=seed,
        train_gene_set_size=train_gene_set_size,
        train=list(set2conditions["train"]),
        val=list(set2conditions["val"]),
        test=list(set2conditions["test"]),
        extra={"generator": "gears", "gears_name": gears_name},
    )

    print(f"==> wrote canonical split to {out}")
    print(f"    train: {len(set2conditions['train'])} perturbations")
    print(f"    val:   {len(set2conditions['val'])} perturbations")
    print(f"    test:  {len(set2conditions['test'])} perturbations")
    return out


def main() -> None:
    """CLI entry point for `python -m scripts.data.gears`."""
    p = argparse.ArgumentParser(
        description="Materialise a gears-sourced dataset split as canonical JSON"
    )
    p.add_argument("--dataset", required=True, help="dataset name under data/")
    p.add_argument("--split-type", default="simulation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-gene-set-size", type=float, default=0.75)
    args = p.parse_args()

    manifest = Manifest.load(args.dataset)
    materialise(
        manifest,
        split_type=args.split_type,
        seed=args.seed,
        train_gene_set_size=args.train_gene_set_size,
    )


if __name__ == "__main__":
    main()
