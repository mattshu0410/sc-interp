"""
Extract per-gene-position hidden-state activations from scGPT for every
transformer layer, for all cells in a requested split of a dataset.

Usage:
    source models/scgpt/.venv/bin/activate
    python -m scripts.run extract-scgpt \
        --dataset norman \
        --split test \
        --hf-repo <your-org>/scgpt-activations-norman \
        --dtype float16

Storage note
------------
Norman has 91 205 cells, ~800 expressed genes/cell on average, 12 layers,
512-dim hidden states:

    all cells  @ float32 → ~1.8 TB
    test split @ float32 → ~197 GB
    test split @ float16 → ~98 GB

Start with --split test --dtype float16 --max-cells N for smoke-tests.

Output layout (local + HF dataset repo)
----------------------------------------
activations/<dataset>_scgpt/
    layer_00_activations.npy   # (total_positions, d_model)  float32 or float16
    layer_00_gene_ids.npy      # (total_positions,)           int32  — vocab token id
    layer_00_cell_ids.npy      # (total_positions,)           int32  — index into cell list
    ...
    layer_11_*
    extraction_metadata.json
    upload_progress.csv        # one row per layer; also pushed to HF after each upload
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

warnings.filterwarnings("ignore", message="flash_attn is not installed")

from scripts import hf
from scripts.extractor import ExtractSpec
from scripts.manifest import Manifest
from scripts.run_scgpt import DEFAULT_PRETRAINED, REPO_ROOT, build_vocab

# ── Constants ─────────────────────────────────────────────────────────────────

ACT_ROOT = REPO_ROOT / "activations"
PROGRESS_CSV = "upload_progress.csv"
METADATA_JSON = "extraction_metadata.json"
CHECKPOINT_JSON = "extraction_checkpoint.json"


# ── CLI args ──────────────────────────────────────────────────────────────────


def _add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--pretrained-dir",
        type=Path,
        default=DEFAULT_PRETRAINED,
        help="scGPT foundation checkpoint folder (needs best_model.pt, args.json, vocab.json)",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=1200,
        help="max gene tokens per cell (truncates to top expressed genes)",
    )
    p.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="flush memmaps and write local checkpoint every N cells",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float16"],
        default="float32",
        help="storage dtype for activation arrays (float16 halves disk usage)",
    )
    p.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="cap number of cells extracted (useful for smoke-tests)",
    )


# ── Load inputs ───────────────────────────────────────────────────────────────


@dataclass
class ScgptExtractInputs:
    pert_data: Any          # gears.PertData
    vocab: Any              # scgpt GeneVocab
    gene_names: np.ndarray  # gene symbols aligned with adata.var


def _load_inputs(manifest: Manifest, args: argparse.Namespace) -> ScgptExtractInputs:
    from gears import PertData

    print("==> loading dataset via GEARS...")
    pert_data = PertData(str(REPO_ROOT / "data"))
    pert_data.load(data_name=manifest.raw["gears_name"])
    pert_data.prepare_split(
        split=manifest.raw.get("split", {}).get("default", "simulation"),
        seed=args.seed,
    )

    vocab = build_vocab(args.pretrained_dir / "vocab.json")

    gene_symbol_col = manifest.var.gene_symbol_column if manifest.var else None
    if gene_symbol_col and gene_symbol_col in pert_data.adata.var.columns:
        gene_names = pert_data.adata.var[gene_symbol_col].values
    else:
        gene_names = pert_data.adata.var_names.values

    n_in_vocab = sum(1 for g in gene_names if g in vocab)
    print(f"==> {n_in_vocab}/{len(gene_names)} genes in scGPT vocab")

    return ScgptExtractInputs(
        pert_data=pert_data,
        vocab=vocab,
        gene_names=gene_names,
    )


# ── Load model ────────────────────────────────────────────────────────────────


@dataclass
class ScgptExtractModel:
    raw: Any        # TransformerModel (nn.Module)
    nns: Any        # NNsight-wrapped model
    device: torch.device
    n_layers: int
    d_model: int
    pad_token_id: int


def _load_model(inputs: ScgptExtractInputs, args: argparse.Namespace) -> ScgptExtractModel:
    from nnsight import NNsight
    from scgpt.model.model import TransformerModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> device: {device}")

    with open(args.pretrained_dir / "args.json") as f:
        margs = json.load(f)

    vocab = inputs.vocab
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=margs["embsize"],
        nhead=margs["nheads"],
        d_hid=margs["d_hid"],
        nlayers=margs["nlayers"],
        vocab=vocab,
        dropout=margs.get("dropout", 0.0),
        pad_token="<pad>",
        pad_value=-2,
        input_emb_style="continuous",
        use_fast_transformer=False,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        cell_emb_style="avg-pool",
        n_cls=1,
    )

    print(f"==> loading checkpoint from {args.pretrained_dir / 'best_model.pt'}")
    raw_ckpt = torch.load(args.pretrained_dir / "best_model.pt", map_location="cpu")
    # unwrap common checkpoint wrapper keys
    if isinstance(raw_ckpt, dict):
        state_dict = (
            raw_ckpt.get("model_state_dict")
            or raw_ckpt.get("state_dict")
            or raw_ckpt.get("model")
            or raw_ckpt
        )
    else:
        state_dict = raw_ckpt

    # FlashMHA uses "Wqkv.*"; standard nn.MultiheadAttention uses "in_proj_*"
    converted = {k.replace("Wqkv.", "in_proj_"): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(converted, strict=False)
    if missing:
        print(f"==> checkpoint missing {len(missing)} keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"==> checkpoint unexpected {len(unexpected)} keys (first 3: {unexpected[:3]})")

    model.to(device).eval()
    n_layers = len(model.transformer_encoder.layers)
    d_model = margs["embsize"]
    print(f"==> model ready: {n_layers}L x {margs['nheads']}H x {d_model}D")

    nns = NNsight(model)

    return ScgptExtractModel(
        raw=model,
        nns=nns,
        device=device,
        n_layers=n_layers,
        d_model=d_model,
        pad_token_id=int(inputs.vocab["<pad>"]),
    )


# ── Tokenisation ──────────────────────────────────────────────────────────────


def _tokenize_cell(
    expr: np.ndarray,
    gene_names: np.ndarray,
    vocab: Any,
    pad_token_id: int,
    max_seq_len: int,
) -> dict | None:
    """Convert a raw expression vector into scGPT token inputs.

    Returns None if no expressed genes map to the vocab.
    Token order: sorted descending by expression (scGPT convention).
    """
    nonzero_idx = np.where(expr > 0)[0]
    if len(nonzero_idx) == 0:
        return None

    token_ids, values, names = [], [], []
    for idx in nonzero_idx:
        gname = gene_names[idx]
        if gname in vocab:
            token_ids.append(vocab[gname])
            values.append(float(expr[idx]))
            names.append(gname)

    if not token_ids:
        return None

    token_ids = np.array(token_ids, dtype=np.int64)
    values = np.array(values, dtype=np.float32)
    order = np.argsort(-values)
    token_ids, values = token_ids[order], values[order]

    if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]
        values = values[:max_seq_len]

    n_genes = len(token_ids)
    pad_len = max_seq_len - n_genes
    gene_ids_padded = np.pad(token_ids, (0, pad_len), constant_values=pad_token_id)
    gene_vals_padded = np.pad(values, (0, pad_len), constant_values=-2.0)
    mask = np.zeros(max_seq_len, dtype=bool)
    mask[n_genes:] = True

    return {
        "gene_ids": gene_ids_padded,
        "gene_values": gene_vals_padded,
        "src_key_padding_mask": mask,
        "n_genes": n_genes,
        "gene_token_ids": token_ids.astype(np.int32),
    }


# ── Progress CSV helpers ───────────────────────────────────────────────────────


def _read_progress(csv_path: Path) -> set[int]:
    """Return set of layer indices already marked 'done' in the CSV."""
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "done":
                done.add(int(row["layer"]))
    return done


def _append_progress(csv_path: Path, layer: int, cells: int, positions: int) -> None:
    """Append a 'done' row for the given layer."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["layer", "cells", "positions", "status", "timestamp"]
        )
        if write_header:
            w.writeheader()
        w.writerow(
            {
                "layer": layer,
                "cells": cells,
                "positions": positions,
                "status": "done",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        )


# ── Main extraction loop ───────────────────────────────────────────────────────


def _extract(
    model: ScgptExtractModel,
    inputs: ScgptExtractInputs,
    args: argparse.Namespace,
) -> None:
    pert_data = inputs.pert_data
    adata = pert_data.adata
    vocab = inputs.vocab
    gene_names = inputs.gene_names
    np_dtype = np.float16 if args.dtype == "float16" else np.float32

    # ── Resolve cell indices for the requested split ──────────────────────────
    pert_col = adata.obs.columns[
        adata.obs.columns.str.lower() == "condition"
    ][0] if "condition" in adata.obs.columns else adata.obs.columns[0]

    split_conditions = set(pert_data.set2conditions.get(args.split, []))
    if not split_conditions:
        raise ValueError(
            f"No conditions found for split={args.split!r}. "
            f"Available: {list(pert_data.set2conditions)}"
        )

    cell_mask = adata.obs[pert_col].isin(split_conditions)
    cell_indices = np.where(cell_mask)[0]
    if args.max_cells is not None:
        cell_indices = cell_indices[: args.max_cells]

    n_cells = len(cell_indices)
    print(
        f"==> split={args.split!r}: {n_cells} cells "
        f"({len(split_conditions)} conditions)"
    )

    # ── Output directory ──────────────────────────────────────────────────────
    out_dir = ACT_ROOT / f"{args.dataset}_scgpt"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Pre-tokenise all cells ────────────────────────────────────────────────
    print("==> tokenising cells...")
    t0 = time.time()
    all_tok: list[dict] = []
    valid_cell_indices: list[int] = []
    X = adata.X

    for ci, adata_idx in enumerate(cell_indices):
        row = X[adata_idx]
        if hasattr(row, "toarray"):
            expr = np.asarray(row.toarray()).ravel()
        else:
            expr = np.asarray(row).ravel()

        tok = _tokenize_cell(
            expr, gene_names, vocab, model.pad_token_id, args.max_seq_len
        )
        if tok is not None:
            all_tok.append(tok)
            valid_cell_indices.append(adata_idx)

        if (ci + 1) % 1000 == 0:
            print(f"    tokenised {ci + 1}/{n_cells} cells...")

    n_valid = len(all_tok)
    total_positions = sum(t["n_genes"] for t in all_tok)
    print(
        f"==> tokenised {n_valid}/{n_cells} cells | "
        f"{total_positions:,} total gene positions | "
        f"{time.time() - t0:.1f}s"
    )

    # ── Try to resume from HF / local checkpoint ──────────────────────────────
    progress_path = out_dir / PROGRESS_CSV
    checkpoint_path = out_dir / CHECKPOINT_JSON

    if args.hf_repo:
        hf.try_download_dataset(args.hf_repo, out_dir, [PROGRESS_CSV])

    layers_done = _read_progress(progress_path)

    start_cell = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_cell = int(ckpt.get("cells_done", 0))
        if start_cell > 0:
            print(f"==> resuming extraction from cell {start_cell}")

    # ── Allocate memmaps for layers not yet uploaded ──────────────────────────
    n_layers = model.n_layers
    d_model = model.d_model
    layers_to_extract = [i for i in range(n_layers) if i not in layers_done]

    if not layers_to_extract:
        print("==> all layers already uploaded, nothing to extract")
        return

    act_maps: dict[int, np.ndarray] = {}
    gid_maps: dict[int, np.ndarray] = {}
    cid_maps: dict[int, np.ndarray] = {}

    mode = "w+" if start_cell == 0 else "r+"
    for layer in layers_to_extract:
        act_path = out_dir / f"layer_{layer:02d}_activations.npy"
        gid_path = out_dir / f"layer_{layer:02d}_gene_ids.npy"
        cid_path = out_dir / f"layer_{layer:02d}_cell_ids.npy"
        act_maps[layer] = np.lib.format.open_memmap(
            act_path, mode=mode, dtype=np_dtype, shape=(total_positions, d_model)
        )
        gid_maps[layer] = np.lib.format.open_memmap(
            gid_path, mode=mode, dtype=np.int32, shape=(total_positions,)
        )
        cid_maps[layer] = np.lib.format.open_memmap(
            cid_path, mode=mode, dtype=np.int32, shape=(total_positions,)
        )

    est_gb = total_positions * d_model * (2 if args.dtype == "float16" else 4) * len(layers_to_extract) / 1e9
    print(f"==> estimated storage for remaining layers: {est_gb:.1f} GB")

    # ── Compute write offset for resume ──────────────────────────────────────
    write_offset = sum(all_tok[ci]["n_genes"] for ci in range(start_cell))
    pos_written = write_offset

    # ── Extraction loop ───────────────────────────────────────────────────────
    print(f"==> extracting activations (cells {start_cell}..{n_valid - 1})...")
    t0 = time.time()
    raw_model = model.raw
    nns = model.nns
    device = model.device

    for ci in range(start_cell, n_valid):
        tok = all_tok[ci]
        n_genes = tok["n_genes"]

        gene_ids_t = (
            torch.tensor(tok["gene_ids"], dtype=torch.long).unsqueeze(0).to(device)
        )
        gene_vals_t = (
            torch.tensor(tok["gene_values"], dtype=torch.float32).unsqueeze(0).to(device)
        )
        mask_t = (
            torch.tensor(tok["src_key_padding_mask"], dtype=torch.bool).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            with nns.trace(
                gene_ids_t,
                values=gene_vals_t,
                src_key_padding_mask=mask_t,
                fn=raw_model._encode,
            ):
                layer_saves = [
                    layer.output.save()
                    for layer in nns.transformer_encoder.layers
                ]

        gene_token_ids = tok["gene_token_ids"]  # (n_genes,) int32

        for layer_idx in layers_to_extract:
            # TransformerEncoderLayer output: (seq_len, batch=1, d_model)
            hidden = layer_saves[layer_idx].value  # tensor
            gene_hidden = hidden[:n_genes, 0, :].cpu().to(
                torch.float16 if args.dtype == "float16" else torch.float32
            ).numpy()
            act_maps[layer_idx][pos_written : pos_written + n_genes] = gene_hidden
            gid_maps[layer_idx][pos_written : pos_written + n_genes] = gene_token_ids
            cid_maps[layer_idx][pos_written : pos_written + n_genes] = ci

        pos_written += n_genes

        if (ci + 1) % args.checkpoint_every == 0 or ci == n_valid - 1:
            for layer_idx in layers_to_extract:
                act_maps[layer_idx].flush()
                gid_maps[layer_idx].flush()
                cid_maps[layer_idx].flush()
            with open(checkpoint_path, "w") as f:
                json.dump(
                    {
                        "cells_done": ci + 1,
                        "positions_written": int(pos_written),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    f,
                    indent=2,
                )

        if (ci + 1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (ci + 1 - start_cell) / elapsed
            eta_min = (n_valid - ci - 1) / max(rate, 1e-6) / 60
            print(
                f"    cell {ci + 1:>6d}/{n_valid} | "
                f"positions {pos_written:>9,} | "
                f"{rate:.1f} cells/s | ETA {eta_min:.1f} min"
            )

    extract_time = time.time() - t0
    print(f"==> extraction done: {pos_written:,} positions in {extract_time:.1f}s")

    # ── Write metadata ────────────────────────────────────────────────────────
    metadata = {
        "model": "scGPT-whole-human",
        "dataset": args.dataset,
        "split": args.split,
        "dtype": args.dtype,
        "architecture": {"n_layers": n_layers, "d_model": d_model},
        "n_cells": n_valid,
        "total_positions": int(pos_written),
        "mean_genes_per_cell": pos_written / max(n_valid, 1),
        "extraction_time_s": extract_time,
        "layers_extracted": layers_to_extract,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    # ── Incremental HF upload ─────────────────────────────────────────────────
    if not args.hf_repo:
        print("==> no --hf-repo specified, skipping upload")
        print(f"==> activations saved locally at {out_dir}")
        return

    print(f"==> uploading to hf dataset:{args.hf_repo} ...")
    for layer_idx in layers_to_extract:
        layer_files = [
            f"layer_{layer_idx:02d}_activations.npy",
            f"layer_{layer_idx:02d}_gene_ids.npy",
            f"layer_{layer_idx:02d}_cell_ids.npy",
        ]
        print(f"    uploading layer {layer_idx:02d}...")
        hf.try_upload_dataset(args.hf_repo, out_dir, layer_files)
        _append_progress(progress_path, layer_idx, n_valid, int(pos_written))
        # Re-push CSV after every layer so a restart knows what's done
        hf.try_upload_dataset(args.hf_repo, out_dir, [PROGRESS_CSV])

    hf.try_upload_dataset(args.hf_repo, out_dir, [METADATA_JSON])
    print(f"==> all done. activations at hf dataset:{args.hf_repo}")


# ── Spec ──────────────────────────────────────────────────────────────────────

SPEC = ExtractSpec(
    name="extract-scgpt",
    add_args=_add_args,
    load_inputs=_load_inputs,
    load_model=_load_model,
    extract=_extract,
)

if __name__ == "__main__":
    from scripts.extractor import extract

    extract(SPEC)
