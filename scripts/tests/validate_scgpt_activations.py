"""
Validate extracted scGPT activations against the source model.

Runs three checks against a local extraction in data/activations/<dataset>_scgpt/:

1. Structural — shape, dtype, no NaNs, referential integrity of gene_ids
   and cell_ids across all layers.
2. Distinctness — different layers produce different activations (catches
   silent "extracted the same layer 12 times" bugs).
3. Reconstruction — re-run the forward pass on a single cell and compare
   the fresh layer outputs to the stored slice, element-wise, within
   float16 precision.

Usage:
    source models/scgpt/.venv/bin/activate
    python -m scripts.validate_scgpt_activations --dataset norman --split test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.extract_scgpt import (
    ACT_ROOT,
    METADATA_JSON,
    _load_inputs,
    _load_model,
    _tokenize_cell,
)
from scripts.manifest import Manifest
from scripts.run_scgpt import DEFAULT_PRETRAINED


def _structural_checks(out_dir: Path, meta: dict) -> None:
    """Shape/dtype/NaN + cross-layer consistency of gene_ids and cell_ids."""
    print("==> [1/3] structural checks")
    n_layers = meta["architecture"]["n_layers"]
    d_model = meta["architecture"]["d_model"]
    n_pos = meta["total_positions"]
    n_cells = meta["n_cells"]
    expected_dtype = np.float16 if meta["dtype"] == "float16" else np.float32

    gid_ref = np.load(out_dir / "layer_00_gene_ids.npy")
    cid_ref = np.load(out_dir / "layer_00_cell_ids.npy")

    assert gid_ref.shape == (n_pos,), f"gene_ids shape {gid_ref.shape} != ({n_pos},)"
    assert cid_ref.shape == (n_pos,), f"cell_ids shape {cid_ref.shape} != ({n_pos},)"
    assert cid_ref.min() >= 0 and cid_ref.max() < n_cells, \
        f"cell_ids out of range: [{cid_ref.min()}, {cid_ref.max()}]"
    assert gid_ref.min() >= 0, f"gene_ids contains negative: min={gid_ref.min()}"

    for layer in range(n_layers):
        act = np.load(out_dir / f"layer_{layer:02d}_activations.npy", mmap_mode="r")
        gid = np.load(out_dir / f"layer_{layer:02d}_gene_ids.npy")
        cid = np.load(out_dir / f"layer_{layer:02d}_cell_ids.npy")

        assert act.shape == (n_pos, d_model), \
            f"layer {layer}: act shape {act.shape} != ({n_pos}, {d_model})"
        assert act.dtype == expected_dtype, \
            f"layer {layer}: dtype {act.dtype} != {expected_dtype}"
        # Spot-check a slice for NaN/Inf (full scan is slow on big memmaps)
        sample = np.asarray(act[:1000])
        assert not np.isnan(sample).any(), f"layer {layer}: NaN in first 1000 rows"
        assert not np.isinf(sample).any(), f"layer {layer}: Inf in first 1000 rows"
        assert np.array_equal(gid, gid_ref), f"layer {layer}: gene_ids drift"
        assert np.array_equal(cid, cid_ref), f"layer {layer}: cell_ids drift"

    print(f"    ok: {n_layers} layers x {n_pos:,} positions x {d_model}d {expected_dtype}")


def _distinctness_check(out_dir: Path, meta: dict) -> None:
    """Different layers should produce materially different activations."""
    print("==> [2/3] distinctness checks")
    n_layers = meta["architecture"]["n_layers"]
    a0 = np.load(out_dir / "layer_00_activations.npy", mmap_mode="r")[:200].astype(np.float32)
    a_last = np.load(out_dir / f"layer_{n_layers - 1:02d}_activations.npy", mmap_mode="r")[:200].astype(np.float32)
    diff = np.linalg.norm(a0 - a_last) / (np.linalg.norm(a0) + 1e-8)
    assert diff > 0.1, f"first and last layer are suspiciously close: rel diff {diff:.4f}"
    print(f"    ok: rel L2 diff layer 0 vs layer {n_layers - 1}: {diff:.3f}")


def _reconstruction_check(
    out_dir: Path,
    meta: dict,
    args: argparse.Namespace,
    cell_position: int = 0,
) -> None:
    """Re-run forward pass on one cell and compare to stored activations."""
    print("==> [3/3] reconstruction check (loads model + dataset)")
    manifest = Manifest.load(args.dataset)

    # _load_inputs needs seed + split on args; extract_scgpt's CLI adds these
    # but validate has its own CLI, so just set sensible defaults here.
    ns = argparse.Namespace(**vars(args))
    if not hasattr(ns, "seed"):
        ns.seed = 42

    inputs = _load_inputs(manifest, ns)
    model = _load_model(inputs, ns)

    # Find which adata row corresponds to cell_id=cell_position for our split
    pert_col = inputs.pert_data.adata.obs.columns[0]
    if "condition" in inputs.pert_data.adata.obs.columns:
        pert_col = "condition"
    split_conditions = set(inputs.pert_data.set2conditions.get(args.split, []))
    cell_mask = inputs.pert_data.adata.obs[pert_col].isin(split_conditions)
    cell_indices = np.where(cell_mask)[0]
    adata_idx = cell_indices[cell_position]

    # Tokenise that cell exactly like extraction does
    row = inputs.pert_data.adata.X[adata_idx]
    expr = np.asarray(row.toarray()).ravel() if hasattr(row, "toarray") else np.asarray(row).ravel()
    tok = _tokenize_cell(expr, inputs.gene_names, inputs.vocab, model.pad_token_id, args.max_seq_len)
    assert tok is not None, "cell produced no tokens"

    # Re-run forward with nnsight, save every layer output
    gene_ids_t = torch.tensor(tok["gene_ids"], dtype=torch.long).unsqueeze(0).to(model.device)
    gene_vals_t = torch.tensor(tok["gene_values"], dtype=torch.float32).unsqueeze(0).to(model.device)
    mask_t = torch.tensor(tok["src_key_padding_mask"], dtype=torch.bool).unsqueeze(0).to(model.device)

    with torch.no_grad():
        with model.nns.trace(gene_ids_t, values=gene_vals_t, src_key_padding_mask=mask_t):
            layer_saves = [layer.output.save() for layer in model.nns.transformer_encoder.layers]

    # Resolve the stored positions for this cell
    cid = np.load(out_dir / "layer_00_cell_ids.npy")
    stored_mask = cid == cell_position
    assert stored_mask.any(), f"cell_id={cell_position} not found in stored activations"
    stored_slice = np.where(stored_mask)[0]
    stored_gene_ids = np.load(out_dir / "layer_00_gene_ids.npy")[stored_slice]

    # Stored gene_ids must match the token ids we just re-tokenised
    assert np.array_equal(stored_gene_ids, tok["gene_token_ids"]), \
        "stored gene_ids for this cell don't match a fresh tokenisation"

    atol = 5e-3 if meta["dtype"] == "float16" else 1e-5
    n_genes = tok["n_genes"]

    for layer in range(model.n_layers):
        hidden = layer_saves[layer].value
        if hidden.is_nested:
            hidden = hidden.to_padded_tensor(0.0)
        fresh = hidden[:n_genes, 0, :].cpu().to(torch.float32).numpy()

        stored = np.asarray(
            np.load(out_dir / f"layer_{layer:02d}_activations.npy", mmap_mode="r")[stored_slice]
        ).astype(np.float32)

        if not np.allclose(fresh, stored, atol=atol, rtol=1e-2):
            max_abs = np.max(np.abs(fresh - stored))
            raise AssertionError(
                f"layer {layer}: reconstruction mismatch, max |Δ|={max_abs:.4e} > {atol:.1e}"
            )

    print(f"    ok: all {model.n_layers} layers reconstructed within atol={atol}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--pretrained-dir", type=Path, default=DEFAULT_PRETRAINED)
    p.add_argument("--max-seq-len", type=int, default=1200)
    p.add_argument("--cell-position", type=int, default=0, help="index into split")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None,
                   help="suffix matching the extract run's --tag")
    p.add_argument("--skip-reconstruction", action="store_true",
                   help="only run structural + distinctness checks (fast)")
    args = p.parse_args()

    dir_name = f"{args.dataset}_scgpt"
    if args.tag:
        dir_name = f"{dir_name}_{args.tag}"
    out_dir = ACT_ROOT / dir_name
    meta_path = out_dir / METADATA_JSON
    if not meta_path.exists():
        raise FileNotFoundError(f"no extraction found at {out_dir}")
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"==> validating {out_dir}")
    print(f"    model={meta['model']} dataset={meta['dataset']} "
          f"split={meta['split']} dtype={meta['dtype']}")
    print(f"    n_cells={meta['n_cells']} n_positions={meta['total_positions']:,}")

    _structural_checks(out_dir, meta)
    _distinctness_check(out_dir, meta)
    if not args.skip_reconstruction:
        _reconstruction_check(out_dir, meta, args, cell_position=args.cell_position)

    print("==> validation passed")


if __name__ == "__main__":
    main()
