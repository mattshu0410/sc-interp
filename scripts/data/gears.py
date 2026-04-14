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
from pathlib import Path

from scripts.data.splits import write_split
from scripts.manifest import REPO_ROOT, Manifest


def load_pert_data(
    manifest: Manifest,
    split_type: str = "simulation",
    seed: int = 42,
) -> "PertData":
    """Load and prepare a GEARS PertData for a manifest whose source is 'gears'.

    Shared entry point for run_scgpt and extract_scgpt. Returns the PertData
    after load + prepare_split. Callers add dataloaders or gene-name lookups
    themselves as needed.

    The gears import is lazy so this module is safe to import from venvs that
    lack gears (only calling this function requires it).
    """
    from gears import PertData

    gears_name = manifest.raw["gears_name"]
    default_split = manifest.raw.get("split", {}).get("default", split_type)

    print("==> loading dataset via GEARS...")
    pert_data = PertData(str(REPO_ROOT / "data"))
    pert_data.load(data_name=gears_name)
    pert_data.prepare_split(split=default_split, seed=seed)
    return pert_data


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

    # GEARS skips its download if the dataset directory already exists, even when
    # only repo-tracked files (e.g. manifest.yaml) are present.  Work around this
    # by temporarily hiding the directory so GEARS fetches the zip from Dataverse.
    dataset_dir = data_dir / gears_name
    h5ad_path = dataset_dir / "perturb_processed.h5ad"
    stashed: dict[str, bytes] = {}
    if dataset_dir.exists() and not h5ad_path.exists():
        print(f"    dataset dir exists but h5ad missing — stashing repo files so GEARS will download")
        for p in list(dataset_dir.iterdir()):
            stashed[p.name] = p.read_bytes()
            p.unlink()
        dataset_dir.rmdir()

    pert_data = PertData(str(data_dir))
    pert_data.load(data_name=gears_name)

    # Restore any stashed repo files (e.g. manifest.yaml)
    for name, content in stashed.items():
        (dataset_dir / name).write_bytes(content)
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
