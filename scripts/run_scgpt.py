"""
Run scGPT perturbation prediction on a dataset from the data/ directory.

Usage:
    source models/scgpt/.venv/bin/activate
    python scripts/run_scgpt.py \\
        --dataset norman \\
        --checkpoint-dir /path/to/scGPT_human \\
        --split test

Checkpoint dir must contain args.json, vocab.json, best_model.pt. The
"whole-human" release lives at
https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y

Follows the eval_perturb recipe from scGPT's Tutorial_Perturbation.ipynb.
"""

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torchtext

torchtext.disable_torchtext_deprecation_warning()
warnings.filterwarnings("ignore", message="flash_attn is not installed")

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from gears import PertData

from scgpt.model import TransformerGenerator
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import set_seed

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Dataset source dispatch ──────────────────────────────────────────────────
# Registry of data source handlers. To support a new manifest source:
#   1. Write a function with signature
#        (manifest: dict, args: Namespace) -> ScgptInputs
#   2. Add it to LOADERS below keyed by the manifest source string.
#
# scGPT's pred_perturb REQUIRES GEARS-style batches: torch_geometric Data
# objects with .pert (list of perturbation labels), .x[:, 0] = ori gene
# values, .x[:, 1] = pert flags, and .y = ground truth expression. New
# sources must either use gears.PertData or produce compatible objects.


@dataclass
class ScgptInputs:
    """Everything run_scgpt needs after dataset-specific loading."""

    loader: Iterable  # yields torch_geometric batches with .pert, .x, .y
    var: pd.DataFrame  # gene metadata aligned with model output columns


def _load_gears(manifest: dict, args: argparse.Namespace) -> ScgptInputs:
    pert_data = PertData(str(REPO_ROOT / "data"))
    pert_data.load(data_name=manifest["gears_name"])
    pert_data.prepare_split(
        split=manifest.get("split", {}).get("default", args.split_type),
        seed=args.seed,
    )
    pert_data.get_dataloader(
        batch_size=args.batch_size, test_batch_size=args.eval_batch_size
    )
    loader = pert_data.dataloader[f"{args.split}_loader"]
    return ScgptInputs(loader=loader, var=pert_data.adata.var.copy())


LOADERS: dict[str, Callable[[dict, argparse.Namespace], ScgptInputs]] = {
    "gears": _load_gears,
}


def load_dataset(manifest: dict, args: argparse.Namespace) -> ScgptInputs:
    source = manifest["source"]
    if source not in LOADERS:
        raise NotImplementedError(
            f"manifest source {source!r} not supported by run_scgpt. "
            f"registered: {sorted(LOADERS)}. "
            f"see LOADERS comment in this file to add a new one."
        )
    return LOADERS[source](manifest, args)


# ── Manifest ──────────────────────────────────────────────────────────────────


def load_manifest(dataset: str) -> dict:
    manifest_path = REPO_ROOT / "data" / dataset / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    with open(manifest_path) as f:
        return yaml.safe_load(f)


# ── Model loading ─────────────────────────────────────────────────────────────


def load_model(
    checkpoint_dir: Path,
    var: pd.DataFrame,
    device: torch.device,
) -> tuple[TransformerGenerator, np.ndarray]:
    vocab_file = checkpoint_dir / "vocab.json"
    args_file = checkpoint_dir / "args.json"
    model_file = checkpoint_dir / "best_model.pt"
    for f in (vocab_file, args_file, model_file):
        if not f.exists():
            raise FileNotFoundError(f"checkpoint file missing: {f}")

    vocab = GeneVocab.from_file(str(vocab_file))
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)

    with open(args_file) as f:
        margs = json.load(f)

    genes = var["gene_name"].tolist()
    var["id_in_vocab"] = [1 if g in vocab else -1 for g in genes]
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int
    )
    n_in_vocab = int((var["id_in_vocab"] == 1).sum())
    print(f"==> {n_in_vocab}/{len(genes)} genes found in scGPT vocab")

    model = TransformerGenerator(
        ntoken=len(vocab),
        d_model=margs["embsize"],
        nhead=margs["nheads"],
        d_hid=margs["d_hid"],
        nlayers=margs["nlayers"],
        nlayers_cls=margs.get("n_layers_cls", 3),
        n_cls=1,
        vocab=vocab,
        dropout=margs.get("dropout", 0.0),
        pad_token="<pad>",
        pad_value=margs.get("pad_value", 0),
        pert_pad_id=margs.get("pert_pad_id", 2),
        use_fast_transformer=False,
    )

    pretrained = torch.load(model_file, map_location=device)
    prefixes = ("encoder", "value_encoder", "transformer_encoder")
    load_dict = {
        k: v for k, v in pretrained.items() if any(k.startswith(p) for p in prefixes)
    }
    model_dict = model.state_dict()
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    print(f"==> loaded {len(load_dict)}/{len(pretrained)} pretrained params")
    return model, gene_ids


# ── Inference ─────────────────────────────────────────────────────────────────


@torch.no_grad()
def predict(
    model: TransformerGenerator,
    loader: Iterable,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
) -> dict:
    pert_cat: list[str] = []
    preds: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []

    for batch in loader:
        batch.to(device)
        pert_cat.extend(batch.pert)
        p = model.pred_perturb(
            batch, include_zero_gene=include_zero_gene, gene_ids=gene_ids
        )
        preds.extend(p.cpu())
        truths.extend(batch.y.cpu())

    return {
        "pert": np.array(pert_cat),
        "pred": torch.stack(preds).numpy().astype(np.float32),
        "truth": torch.stack(truths).numpy().astype(np.float32),
    }


def save_predictions(
    results: dict,
    var: pd.DataFrame,
    output: Path,
    dataset: str,
    split: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    adata = ad.AnnData(
        X=results["pred"],
        obs=pd.DataFrame({"condition": results["pert"]}),
        var=var.reset_index(drop=False),
        layers={"truth": results["truth"]},
    )
    adata.uns["model"] = "scgpt"
    adata.uns["dataset"] = dataset
    adata.uns["split"] = split
    adata.write_h5ad(output)
    print(f"==> wrote {adata.shape} to {output}")


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset name under data/")
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        type=Path,
        help="scGPT checkpoint folder (args.json, vocab.json, best_model.pt)",
    )
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument(
        "--output",
        type=Path,
        help="output h5ad path, default predictions/scgpt_<dataset>_<split>.h5ad",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument(
        "--include-zero-gene",
        default="batch-wise",
        choices=["all", "batch-wise"],
    )
    p.add_argument(
        "--split-type",
        default="simulation",
        help="gears split type (simulation, combo_seen0, ...), "
        "overridden by manifest.split.default if set",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> device: {device}")

    manifest = load_manifest(args.dataset)
    print(f"==> dataset: {manifest['name']}")

    inputs = load_dataset(manifest, args)
    model, gene_ids = load_model(args.checkpoint_dir, inputs.var, device)

    print(f"==> running inference on {args.split} split")
    results = predict(model, inputs.loader, gene_ids, args.include_zero_gene, device)

    output = args.output or (
        REPO_ROOT / "predictions" / f"scgpt_{args.dataset}_{args.split}.h5ad"
    )
    save_predictions(results, inputs.var, output, args.dataset, args.split)


if __name__ == "__main__":
    main()
