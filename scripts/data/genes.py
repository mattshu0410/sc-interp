"""Gene ID helpers that consume a Manifest's VarSchema."""

from __future__ import annotations

import re

from scripts.manifest import VarSchema

_ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G\d+(\.\d+)?$")


def build_symbol_to_id(adata, var: VarSchema) -> dict[str, str]:
    """Build a gene symbol -> gene ID dict from adata.var per the schema.

    Raises ValueError if the declared columns are missing or if
    gene_id_type is 'ensembl' but the first 20 IDs don't mostly match
    the Ensembl pattern.
    """
    if var.gene_symbol_column is None:
        raise ValueError("manifest.var.gene_symbol_column not declared")
    for col in (var.gene_symbol_column, var.gene_id_column):
        if col is not None and col not in adata.var.columns:
            raise ValueError(
                f"{col!r} not in adata.var columns: {list(adata.var.columns)}"
            )

    ids = (
        adata.var_names.astype(str)
        if var.gene_id_column is None
        else adata.var[var.gene_id_column].astype(str)
    )
    symbols = adata.var[var.gene_symbol_column].astype(str)

    if var.gene_id_type == "ensembl":
        sample = list(ids[:20])
        if sum(_ENSEMBL_RE.match(v) is not None for v in sample) < 15:
            raise ValueError(
                f"gene_id_type=ensembl but var IDs don't match pattern. "
                f"first 20: {sample}"
            )
    elif var.gene_id_type not in ("symbol",):
        raise ValueError(f"unknown gene_id_type: {var.gene_id_type!r}")

    return dict(zip(symbols, ids))
