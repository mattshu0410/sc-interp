from __future__ import annotations

import torch

from scripts.interp.scgpt_inputs import scatter_back


def test_scatter_back_places_values_at_input_gene_ids() -> None:
    torch.manual_seed(0)
    bs, n_genes, k = 3, 10, 4
    input_gene_ids = torch.tensor([0, 3, 5, 9])
    output_values = torch.randn(bs, k)

    pred = scatter_back(output_values, input_gene_ids, n_genes)

    assert pred.shape == (bs, n_genes)
    assert torch.equal(pred[:, input_gene_ids], output_values)


def test_scatter_back_leaves_unselected_columns_zero() -> None:
    bs, n_genes = 2, 8
    input_gene_ids = torch.tensor([1, 4, 7])
    output_values = torch.ones(bs, len(input_gene_ids))

    pred = scatter_back(output_values, input_gene_ids, n_genes)

    unselected = torch.tensor([i for i in range(n_genes) if i not in input_gene_ids])
    assert torch.all(pred[:, unselected] == 0)


def test_scatter_back_wrong_indices_would_misplace_values() -> None:
    # Guards against the easy-to-miss bug where mapped_input_gene_ids (vocab
    # token ids, often much larger than n_genes) gets passed instead of
    # input_gene_ids. Using vocab-style large indices must raise, not silently
    # misindex or pad.
    bs, n_genes = 1, 5
    bad_ids = torch.tensor([0, 1, 999])  # 999 > n_genes
    values = torch.zeros(bs, 3)
    try:
        scatter_back(values, bad_ids, n_genes)
    except IndexError:
        return
    raise AssertionError("expected IndexError for out-of-range column index")