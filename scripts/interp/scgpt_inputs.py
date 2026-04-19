from __future__ import annotations

import torch


def scatter_back(
    output_values: torch.Tensor,
    input_gene_ids: torch.Tensor,
    n_genes: int,
) -> torch.Tensor:
    """Place per-gene predictions into their (batch, n_genes) column slots.

    `output_values` has shape (batch, len(input_gene_ids)); unselected gene
    columns stay zero, matching pred_perturb's behavior. `input_gene_ids` is
    the column index slice — NOT the vocab token ids. Passing the vocab ids
    (mapped_input_gene_ids) here would silently index into wrong columns and
    produce garbage predictions.
    """
    bs = output_values.shape[0]
    pred_full = torch.zeros(bs, n_genes, device=output_values.device)
    pred_full[:, input_gene_ids] = output_values
    return pred_full
