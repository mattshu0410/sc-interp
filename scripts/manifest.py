"""Dataset manifest schema and loader.

Each manifest at data/manifests/<name>.yaml is a loader-agnostic
descriptor: a dispatch key (`source`), the obs columns runners need to
identify perturbations and controls, and an arbitrary `raw` bag for
loader-specific fields. Loaders read their own keys from `raw`. Manifests
live outside data/<name>/ so dataset loaders (e.g. GEARS) can treat
data/<name>/ as a pure download cache.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ObsSchema:
    """obs column names and the control-row label."""
    pert_col: str
    control_label: str
    cell_type_col: str | None = None


@dataclass
class VarSchema:
    """How genes are identified in adata.var.

    `gene_id_type` labels the format of the canonical gene ID ("ensembl",
    "symbol", "entrez"). `gene_id_column` is the name of the column holding
    the IDs, or None to use var_names (the index). `gene_symbol_column`
    points at a companion column holding gene symbols, or None if symbols
    are unavailable or are themselves the canonical IDs.
    """
    gene_id_type: str
    gene_id_column: str | None = None
    gene_symbol_column: str | None = None


@dataclass
class Manifest:
    """Typed view of data/manifests/<name>.yaml."""
    name: str
    source: str
    obs: ObsSchema
    var: VarSchema | None = None
    description: str = ""
    shape: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, name: str, repo_root: Path = REPO_ROOT) -> Manifest:
        """Read data/manifests/<name>.yaml and return a validated Manifest."""
        path = repo_root / "data" / "manifests" / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no manifest at {path}")
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        """Construct from a parsed yaml dict, raising on missing required keys."""
        for key in ("name", "source", "obs"):
            if key not in d:
                raise ValueError(f"manifest missing required key: {key!r}")

        obs_d = d["obs"]
        for key in ("pert_col", "control_label"):
            if key not in obs_d:
                raise ValueError(f"manifest.obs missing required key: {key!r}")

        var_d = d.get("var")
        var: VarSchema | None = None
        if var_d is not None:
            if "gene_id_type" not in var_d:
                raise ValueError("manifest.var missing required key: 'gene_id_type'")
            var = VarSchema(
                gene_id_type=var_d["gene_id_type"],
                gene_id_column=var_d.get("gene_id_column"),
                gene_symbol_column=var_d.get("gene_symbol_column"),
            )

        return cls(
            name=d["name"],
            source=d["source"],
            obs=ObsSchema(
                pert_col=obs_d["pert_col"],
                control_label=obs_d["control_label"],
                cell_type_col=obs_d.get("cell_type_col"),
            ),
            var=var,
            description=d.get("description", ""),
            shape=d.get("shape", {}) or {},
            files=d.get("files", {}) or {},
            raw=d,
        )

    def dataset_dir(self, repo_root: Path = REPO_ROOT) -> Path:
        """Path to data/<name>/."""
        return repo_root / "data" / self.name
