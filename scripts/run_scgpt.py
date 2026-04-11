"""
Fine-tune and run scGPT perturbation prediction on a dataset from data/.

Usage:
    source models/scgpt/.venv/bin/activate
    python scripts/run_scgpt.py \\
        --dataset norman \\
        --split test

If a fine-tuned checkpoint for the dataset already exists at
models/scgpt/checkpoints/<dataset>_ft/, training is skipped. Use
--force-finetune to retrain from scratch.

The pretrained whole-human checkpoint comes from the setup script
(models/setup_scgpt.sh). Our fine-tuning starts from those weights and
early-stops on validation pearson of predicted vs true perturbation
response, matching scGPT's own benchmark protocol.

Training budget is expressed in "cells seen" (total number of training
cells processed across all gradient steps), not epochs or wall-clock.
Early stopping on val pearson fires first in practice, cells-seen is a
ceiling.
"""

import argparse
import copy
import json
import time
import warnings
from dataclasses import asdict, dataclass
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

from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.model.generation_model import map_raw_id_to_vocab_id
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import compute_perturbation_metrics, set_seed

import _hf       # scripts/_hf.py, on sys.path since this file lives in scripts/
import _wandb    # scripts/_wandb.py

FT_FILES = ["best_model.pt", "training_stats.json"]

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO_ROOT / "models" / "scgpt" / "checkpoints"
DEFAULT_PRETRAINED = CKPT_ROOT / "scGPT_human"


# ── Dataset dispatch ──────────────────────────────────────────────────────────
# Registry of manifest-source handlers. To add a new source:
#   1. Write a function (manifest, args) -> ScgptInputs
#   2. Register it in LOADERS below.
#
# scGPT consumes GEARS-style batches (torch_geometric Data with .pert, .x, .y,
# .de_idx), so new sources must produce those, not raw AnnData.


@dataclass
class ScgptInputs:
    pert_data: PertData               # needed later for ctrl adata in metrics
    train_loader: Iterable
    val_loader: Iterable
    test_loader: Iterable
    var: pd.DataFrame


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
    return ScgptInputs(
        pert_data=pert_data,
        train_loader=pert_data.dataloader["train_loader"],
        val_loader=pert_data.dataloader["val_loader"],
        test_loader=pert_data.dataloader["test_loader"],
        var=pert_data.adata.var.copy(),
    )


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


def load_manifest(dataset: str) -> dict:
    manifest_path = REPO_ROOT / "data" / dataset / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    with open(manifest_path) as f:
        return yaml.safe_load(f)


# ── Model construction and checkpoint loading ────────────────────────────────


def build_vocab(vocab_file: Path) -> GeneVocab:
    vocab = GeneVocab.from_file(str(vocab_file))
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    return vocab


def build_gene_ids(var: pd.DataFrame, vocab: GeneVocab) -> np.ndarray:
    genes = var["gene_name"].tolist()
    var["id_in_vocab"] = [1 if g in vocab else -1 for g in genes]
    gene_ids = np.array(
        [vocab[g] if g in vocab else vocab["<pad>"] for g in genes], dtype=int
    )
    n = int((var["id_in_vocab"] == 1).sum())
    print(f"==> {n}/{len(genes)} genes found in scGPT vocab")
    return gene_ids


def build_model(margs: dict, vocab: GeneVocab) -> TransformerGenerator:
    return TransformerGenerator(
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


def load_pretrained_weights(
    model: TransformerGenerator, model_file: Path, device: torch.device
) -> None:
    """Load the whole-human foundation checkpoint into a freshly built model.

    The released weights were saved with fast_transformer=True (flash-attn
    FlashMHA module → Wqkv). Our model uses standard torch MultiheadAttention
    (in_proj_{weight,bias}). The tensor layout is identical, only the key
    names differ, so we rename during load.
    """
    pretrained = torch.load(model_file, map_location=device)
    prefixes = ("encoder", "value_encoder", "transformer_encoder")
    load_dict: dict[str, torch.Tensor] = {}
    for k, v in pretrained.items():
        if not any(k.startswith(p) for p in prefixes):
            continue
        k = k.replace("self_attn.Wqkv.weight", "self_attn.in_proj_weight")
        k = k.replace("self_attn.Wqkv.bias", "self_attn.in_proj_bias")
        load_dict[k] = v
    model_dict = model.state_dict()
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    print(f"==> loaded {len(load_dict)}/{len(pretrained)} pretrained params")


def load_finetuned_weights(
    model: TransformerGenerator, model_file: Path, device: torch.device
) -> None:
    """Load a cached fine-tuned checkpoint saved by this script."""
    state = torch.load(model_file, map_location=device)
    model.load_state_dict(state)
    print(f"==> loaded fine-tuned weights from {model_file}")


# ── Training ──────────────────────────────────────────────────────────────────


@dataclass
class TrainStats:
    num_epochs_trained: int
    cells_seen: int
    steps: int
    wall_clock_s: float
    best_val_metrics: dict         # full compute_perturbation_metrics dict at the best epoch
    best_val_epoch: int
    stop_metric: str               # which key of best_val_metrics drove selection
    reason: str                    # "max_epochs" | "early_stop" | "skipped_cached"
    wandb_run_url: str | None = None


def forward_pass(
    model: TransformerGenerator,
    batch_data,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
    amp: bool,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replicates the body of pred_perturb but keeps gradients enabled.

    Clips the input gene set to max_seq_len via random subsampling when
    include_zero_gene="all", matching the tutorial's training recipe.
    Without this the attention memory blows up on datasets with more
    genes than the model's sequence length (Norman has 5045 genes vs
    max_seq_len 1200).
    """
    batch_size = len(batch_data.y)
    batch_data.to(device)
    x = batch_data.x
    n_genes = len(gene_ids)
    ori_gene_values = x[:, 0].view(batch_size, n_genes)
    pert_flags = x[:, 1].long().view(batch_size, n_genes)
    target_gene_values = batch_data.y

    if include_zero_gene == "all":
        input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
    elif include_zero_gene == "batch-wise":
        input_gene_ids = (
            ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
        )
    else:
        raise ValueError(f"unknown include_zero_gene: {include_zero_gene}")

    if len(input_gene_ids) > max_seq_len:
        perm = torch.randperm(len(input_gene_ids), device=device)[:max_seq_len]
        input_gene_ids = input_gene_ids[perm]

    input_values = ori_gene_values[:, input_gene_ids]
    input_pert_flags = pert_flags[:, input_gene_ids]
    target_values = target_gene_values[:, input_gene_ids]

    mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
    src_key_padding_mask = torch.zeros_like(
        input_values, dtype=torch.bool, device=device
    )

    with torch.cuda.amp.autocast(enabled=amp):
        output_dict = model(
            mapped_input_gene_ids,
            input_values,
            input_pert_flags,
            src_key_padding_mask=src_key_padding_mask,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
        )
    return output_dict["mlm_output"], target_values, input_values


@torch.no_grad()
def evaluate_val(
    model: TransformerGenerator,
    loader: Iterable,
    pert_data: PertData,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
) -> dict:
    """Run the loader through pred_perturb and compute scGPT's own metrics."""
    model.eval()
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
    results = {
        "pert_cat": np.array(pert_cat),
        "pred": torch.stack(preds).numpy().astype(np.float32),
        "truth": torch.stack(truths).numpy().astype(np.float32),
    }
    ctrl_adata = pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    return compute_perturbation_metrics(results, ctrl_adata)


def finetune(
    model: TransformerGenerator,
    inputs: ScgptInputs,
    gene_ids: np.ndarray,
    device: torch.device,
    num_epochs: int,
    early_stop: int,
    stop_metric: str,
    lr: float,
    include_zero_gene: str,
    amp: bool,
    max_seq_len: int,
) -> tuple[TransformerGenerator, TrainStats]:
    """Per-epoch fine-tuning loop matching scGPT Tutorial_Perturbation.ipynb.

    One epoch = one pass over train_loader, followed by eval_perturb +
    compute_perturbation_metrics on val_loader. Best model selected by
    stop_metric (tutorial uses 'pearson'). Early-stop fires after
    early_stop epochs of no improvement on stop_metric.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    best_val_score = -float("inf")
    best_state: dict | None = None
    best_epoch = 0
    best_metrics: dict = {}
    patience = 0
    cells_seen = 0
    steps = 0
    epochs_trained = 0
    t0 = time.time()
    reason = "max_epochs"

    print(
        f"==> fine-tuning up to {num_epochs} epochs, "
        f"early_stop={early_stop}, stop_metric={stop_metric!r}"
    )

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for batch in inputs.train_loader:
            optimizer.zero_grad()
            output, target, inp = forward_pass(
                model, batch, gene_ids, include_zero_gene, device, amp, max_seq_len
            )
            mask = torch.ones_like(inp, dtype=torch.bool)
            loss = masked_mse_loss(output, target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            cells_seen += len(batch.y)
            steps += 1
            epoch_loss += float(loss.item())
            epoch_batches += 1

        metrics = evaluate_val(
            model,
            inputs.val_loader,
            inputs.pert_data,
            gene_ids,
            include_zero_gene,
            device,
        )
        epochs_trained = epoch
        avg_loss = epoch_loss / max(epoch_batches, 1)
        elapsed = time.time() - epoch_start
        print(
            f"    [epoch {epoch:2d}/{num_epochs} | loss {avg_loss:.4f} | "
            f"pearson {metrics.get('pearson', float('nan')):.4f} | "
            f"pearson_delta {metrics.get('pearson_delta', float('nan')):.4f} | "
            f"{elapsed:.1f}s]"
        )
        _wandb.log(
            {
                "train/loss": avg_loss,
                "train/cells_seen": cells_seen,
                "train/epoch": epoch,
                **{f"val/{k}": float(v) for k, v in metrics.items()},
            },
            step=epoch,
        )

        val_score = float(metrics.get(stop_metric, float("nan")))
        if val_score > best_val_score:
            best_val_score = val_score
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            best_metrics = {k: float(v) for k, v in metrics.items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stop:
                reason = "early_stop"
                print(f"==> early stop at epoch {epoch}")
                break

        scheduler.step()

    if best_state is not None:
        model.load_state_dict(best_state)

    stats = TrainStats(
        num_epochs_trained=epochs_trained,
        cells_seen=cells_seen,
        steps=steps,
        wall_clock_s=time.time() - t0,
        best_val_metrics=best_metrics,
        best_val_epoch=best_epoch,
        stop_metric=stop_metric,
        reason=reason,
        wandb_run_url=_wandb.url(),
    )
    print(
        f"==> fine-tune done: reason={reason}, "
        f"best {stop_metric}={best_val_score:.4f} at epoch {best_epoch}, "
        f"cells_seen={cells_seen:,}, wall={stats.wall_clock_s/60:.1f}min"
    )
    return model, stats


def finetune_cache_dir(dataset: str) -> Path:
    return CKPT_ROOT / f"{dataset}_ft"


def maybe_finetune(
    model: TransformerGenerator,
    inputs: ScgptInputs,
    gene_ids: np.ndarray,
    dataset: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[TransformerGenerator, TrainStats]:
    cache_dir = finetune_cache_dir(dataset)
    ckpt_path = cache_dir / "best_model.pt"
    stats_path = cache_dir / "training_stats.json"

    # Priority: local cache > HF hub > train
    if not ckpt_path.exists() and args.hf_repo and not args.force_finetune:
        _hf.try_download(args.hf_repo, cache_dir, FT_FILES)

    if ckpt_path.exists() and not args.force_finetune:
        print(f"==> fine-tuned checkpoint exists at {ckpt_path}, loading")
        load_finetuned_weights(model, ckpt_path, device)
        if stats_path.exists():
            with open(stats_path) as f:
                stats = TrainStats(**json.load(f))
        else:
            stats = TrainStats(
                num_epochs_trained=0,
                cells_seen=0,
                steps=0,
                wall_clock_s=0.0,
                best_val_metrics={},
                best_val_epoch=0,
                stop_metric=args.stop_metric,
                reason="skipped_cached",
            )
        return model, stats

    model, stats = finetune(
        model=model,
        inputs=inputs,
        gene_ids=gene_ids,
        device=device,
        num_epochs=args.num_epochs,
        early_stop=args.early_stop,
        stop_metric=args.stop_metric,
        lr=args.lr,
        include_zero_gene=args.include_zero_gene,
        amp=args.amp,
        max_seq_len=args.max_seq_len,
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"==> saved fine-tuned checkpoint to {ckpt_path}")

    if args.hf_repo:
        _hf.try_upload(args.hf_repo, cache_dir, FT_FILES)

    return model, stats


# ── Inference and saving ─────────────────────────────────────────────────────


@torch.no_grad()
def predict(
    model: TransformerGenerator,
    loader: Iterable,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
) -> dict:
    model.eval()
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
    train_stats: TrainStats,
    ctrl_adata: ad.AnnData,
    pert_col: str,
    control_label: str,
) -> None:
    """Write pred + truth + real controls to one self-contained h5ad.

    Layout:
        .X                = predicted expression (perturbed cells) + real control expression
        .layers["truth"]  = ground truth expression (perturbed cells) + real control expression
        .obs[pert_col]    = perturbation label per row, control_label for control rows

    Downstream eval (cell-eval, stratification) can consume this directly
    without needing the raw dataset file.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    ctrl_X = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, "toarray") else np.asarray(ctrl_adata.X)
    pred_X = np.vstack([results["pred"], ctrl_X])
    truth_X = np.vstack([results["truth"], ctrl_X])
    labels = np.concatenate(
        [results["pert"], np.array([control_label] * ctrl_adata.n_obs)]
    )

    adata = ad.AnnData(
        X=pred_X,
        obs=pd.DataFrame({pert_col: labels}),
        var=var.copy(),
        layers={"truth": truth_X},
    )
    adata.uns["model"] = "scgpt"
    adata.uns["dataset"] = dataset
    adata.uns["split"] = split
    adata.uns["pert_col"] = pert_col
    adata.uns["control_label"] = control_label
    adata.uns["train_stats"] = asdict(train_stats)
    adata.write_h5ad(output)
    print(f"==> wrote {adata.shape} to {output} ({ctrl_adata.n_obs} control cells included)")


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="dataset name under data/")
    p.add_argument(
        "--pretrained-dir",
        type=Path,
        default=DEFAULT_PRETRAINED,
        help="scGPT foundation checkpoint folder",
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
        "--include-zero-gene", default="all", choices=["all", "batch-wise"]
    )
    p.add_argument("--split-type", default="simulation")
    p.add_argument("--seed", type=int, default=42)

    # Fine-tuning knobs (defaults match scGPT Tutorial_Perturbation.ipynb)
    p.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        help="max training epochs, default 15 (tutorial)",
    )
    p.add_argument(
        "--early-stop",
        type=int,
        default=10,
        help="stop after N epochs of no val improvement, default 10 (tutorial)",
    )
    p.add_argument(
        "--stop-metric",
        default="pearson",
        choices=["pearson", "pearson_delta", "pearson_de", "pearson_de_delta"],
        help="which compute_perturbation_metrics key drives best-checkpoint selection",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=1536,
        help="sequence length cap when include_zero_gene=all",
    )
    p.add_argument(
        "--force-finetune",
        action="store_true",
        help="ignore cached fine-tuned checkpoint and retrain",
    )
    p.add_argument(
        "--hf-repo",
        type=str,
        default=None,
        help="HuggingFace repo id (e.g. user/scGPT-norman-ft). If set, try "
        "downloading fine-tuned weights from here when no local cache, "
        "and upload after training.",
    )
    _wandb.add_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> device: {device}")

    manifest = load_manifest(args.dataset)
    print(f"==> dataset: {manifest['name']}")

    _wandb.init("scgpt", args.dataset, args)

    inputs = load_dataset(manifest, args)

    # Build model from foundation architecture
    vocab = build_vocab(args.pretrained_dir / "vocab.json")
    with open(args.pretrained_dir / "args.json") as f:
        margs = json.load(f)
    gene_ids = build_gene_ids(inputs.var, vocab)
    model = build_model(margs, vocab).to(device)
    load_pretrained_weights(model, args.pretrained_dir / "best_model.pt", device)

    # Fine-tune (or load cached fine-tuned weights)
    model, train_stats = maybe_finetune(
        model, inputs, gene_ids, args.dataset, device, args
    )

    # Inference on requested split
    loader = {
        "train": inputs.train_loader,
        "val": inputs.val_loader,
        "test": inputs.test_loader,
    }[args.split]
    print(f"==> running inference on {args.split} split")
    results = predict(model, loader, gene_ids, args.include_zero_gene, device)

    output = args.output or (
        REPO_ROOT / "predictions" / f"scgpt_{args.dataset}_{args.split}.h5ad"
    )
    pert_col = manifest["obs"]["pert_col"]
    control_label = manifest["obs"]["control_label"]
    ctrl_adata = inputs.pert_data.adata[
        inputs.pert_data.adata.obs[pert_col] == control_label
    ]
    save_predictions(
        results,
        inputs.var,
        output,
        args.dataset,
        args.split,
        train_stats,
        ctrl_adata,
        pert_col,
        control_label,
    )

    _wandb.finish()


if __name__ == "__main__":
    main()
