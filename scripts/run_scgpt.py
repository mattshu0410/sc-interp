"""
Fine-tune and run scGPT perturbation prediction on a dataset from data/.

Usage:
    source models/scgpt/.venv/bin/activate
    python -m scripts.run scgpt --dataset norman --split test

If a fine-tuned checkpoint for the dataset already exists at
models/scgpt/checkpoints/<dataset>_ft/, training is skipped. Use
--force to retrain from scratch.

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
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable

warnings.filterwarnings("ignore", message="flash_attn is not installed")

import anndata as ad
import numpy as np
import pandas as pd
import torch

# pandas 3.0 + anndata 0.9 ArrowStringArray compat — must precede gears.
from scripts.data.gears import attach_obs_names_to_pert_data, configure_pandas_for_gears
from scripts.data.state_replogle import load_state_replogle
configure_pandas_for_gears()

from gears import PertData
from nnsight import NNsight
from tqdm import tqdm

from scgpt.loss import masked_mse_loss
from scgpt.model import TransformerGenerator
from scgpt.model.gene_priors import GenePriorEncoder
from scgpt.model.generation_model import map_raw_id_to_vocab_id
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.utils import compute_perturbation_metrics, set_seed

from scripts import wb
from scripts.cache import TrainStats, cache_or_train
from scripts.interp.hook_sinks import H5ActivationSink, default_activation_out
from scripts.interp.hooks import HookManager
from scripts.interp.scgpt_inputs import scatter_back
from scripts.manifest import Manifest
from scripts.runner import RunnerSpec

FT_FILES = ["best_model.pt"]

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO_ROOT / "models" / "scgpt" / "checkpoints"
DEFAULT_PRETRAINED = CKPT_ROOT / "scGPT_human"


# ── Dataset dispatch ──────────────────────────────────────────────────────────
# Registry of manifest-source handlers. Add a new source by writing a
# (manifest, args) -> ScgptInputs function and registering it in LOADERS.
# scGPT consumes GEARS-style torch_geometric batches (.pert, .x, .y, .de_idx)
# so every handler must produce those.


@dataclass
class ScgptInputs:
    """Everything finetune/predict need from the dataset loader.

    Source-neutral: each loader (gears, state_replogle, …) is responsible
    for producing pyg-compatible DataLoaders and the control-cell subset.
    The runner does not touch any source-specific object (PertData, etc.).
    """
    train_loader: Iterable
    val_loader: Iterable
    test_loader: Iterable
    ctrl_adata: ad.AnnData
    var: pd.DataFrame
    sample_loader: Iterable | None = None


def _ensure_legacy_x_layout(pert_data) -> None:
    """Append a per-gene perturbation flag column to each Data.x in place.

    cell-gears 0.1.x stores Data.x as (n_genes, 1) with a Data.pert_idx
    attribute that indexes into pert_data.pert_names (a GO-derived list,
    not the dataset gene axis). scGPT's pred_perturb and this runner's
    build_forward_args read x[:, 1] expecting a 0/1 gene-axis flag (the
    legacy 0.0.x layout). Map pert_idx → pert_names[i] → gene_names.index
    and write the flag column. Idempotent: bails on the first already-2-col
    Data since the layout is uniform across the dict.
    """
    pert_names = list(pert_data.pert_names)
    gene_names = list(pert_data.adata.var["gene_name"])
    pert_to_gene = {
        i: gene_names.index(p)
        for i, p in enumerate(pert_names)
        if p in gene_names
    }
    for plist in pert_data.dataset_processed.values():
        for d in plist:
            if d.x.shape[1] >= 2:
                return
            flags = torch.zeros(d.x.shape[0], 1, dtype=d.x.dtype)
            for p in d.pert_idx:
                gi = pert_to_gene.get(int(p))
                if gi is not None:
                    flags[gi, 0] = 1.0
            d.x = torch.cat([d.x, flags], dim=1)


def _load_gears(manifest: Manifest, args: argparse.Namespace) -> ScgptInputs:
    """Build GEARS dataloaders for a manifest whose source is 'gears'."""
    # default_pert_graph=False: must match scripts/data/gears.py — the cached
    # cell_graphs.pkl stores pert_idx values indexing whichever pert_names was
    # active when it was written. Disagreeing here makes the legacy-x adapter
    # below resolve indices against the wrong list.
    pert_data = PertData(str(REPO_ROOT / "data"), default_pert_graph=False)
    pert_data.load(data_name=manifest.raw["gears_name"])
    _ensure_legacy_x_layout(pert_data)
    attach_obs_names_to_pert_data(pert_data)
    pert_data.prepare_split(
        split=manifest.raw.get("split", {}).get("default", args.split_type),
        seed=args.seed,
    )
    pert_data.get_dataloader(
        batch_size=args.batch_size, test_batch_size=args.eval_batch_size
    )
    pert_col = manifest.obs.pert_col
    control_label = manifest.obs.control_label
    ctrl_adata = pert_data.adata[
        pert_data.adata.obs[pert_col] == control_label
    ].copy()
    return ScgptInputs(
        train_loader=pert_data.dataloader["train_loader"],
        val_loader=pert_data.dataloader["val_loader"],
        test_loader=pert_data.dataloader["test_loader"],
        ctrl_adata=ctrl_adata,
        var=pert_data.adata.var.copy(),
    )


def _load_state_replogle(
    manifest: Manifest, args: argparse.Namespace
) -> ScgptInputs:
    """Build pyg DataLoaders from State-Replogle-Filtered for cross-cell-line eval."""
    (
        train_loader, val_loader, test_loader, ctrl_adata, var, sample_loader,
    ) = load_state_replogle(manifest, args)
    return ScgptInputs(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        ctrl_adata=ctrl_adata,
        var=var,
        sample_loader=sample_loader,
    )


LOADERS: dict[str, Callable[[Manifest, argparse.Namespace], ScgptInputs]] = {
    "gears": _load_gears,
    "state_replogle": _load_state_replogle,
}


def load_dataset(manifest: Manifest, args: argparse.Namespace) -> ScgptInputs:
    """Dispatch to the LOADERS handler for manifest.source."""
    if manifest.source not in LOADERS:
        raise NotImplementedError(
            f"manifest source {manifest.source!r} not supported by run_scgpt. "
            f"registered: {sorted(LOADERS)}"
        )
    return LOADERS[manifest.source](manifest, args)


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


def build_model(
    margs: dict,
    vocab: GeneVocab,
    gene_prior: GenePriorEncoder | None = None,
) -> TransformerGenerator:
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
        gene_prior=gene_prior,
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
class ScgptForwardArgs:
    """Inputs to TransformerGenerator.forward + per-batch slice metadata.

    `input_gene_ids` is the column index slice into the (batch, n_genes)
    expression vector — needed by the predict path to scatter `mlm_output`
    back into a full-width prediction tensor (see pred_perturb).
    """
    mapped_input_gene_ids: torch.Tensor
    input_values: torch.Tensor
    input_pert_flags: torch.Tensor
    src_key_padding_mask: torch.Tensor
    target_values: torch.Tensor
    input_gene_ids: torch.Tensor
    n_genes: int


def build_forward_args(
    batch_data,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
    max_seq_len: int,
) -> ScgptForwardArgs:
    """Prepare per-batch model inputs the way pred_perturb does.

    Random-subsamples the gene axis to max_seq_len when include_zero_gene="all"
    — Norman has 5045 genes vs max_seq_len 1200, attention memory blows up
    otherwise. Each batch sees a different gene window.
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
    return ScgptForwardArgs(
        mapped_input_gene_ids=mapped_input_gene_ids,
        input_values=input_values,
        input_pert_flags=input_pert_flags,
        src_key_padding_mask=src_key_padding_mask,
        target_values=target_values,
        input_gene_ids=input_gene_ids,
        n_genes=n_genes,
    )


def forward_pass(
    model: TransformerGenerator,
    batch_data,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
    amp: bool,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train-time forward: prep inputs, run under AMP, return mlm/target/input."""
    fa = build_forward_args(
        batch_data, gene_ids, include_zero_gene, device, max_seq_len
    )
    with torch.cuda.amp.autocast(enabled=amp):
        output_dict = model(
            fa.mapped_input_gene_ids,
            fa.input_values,
            fa.input_pert_flags,
            src_key_padding_mask=fa.src_key_padding_mask,
            CLS=False,
            CCE=False,
            MVC=False,
            ECS=False,
        )
    return output_dict["mlm_output"], fa.target_values, fa.input_values


@torch.no_grad()
def evaluate_val(
    model: TransformerGenerator,
    loader: Iterable,
    ctrl_adata: ad.AnnData,
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
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimiser, step_size=1, gamma=0.9)
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
            optimiser.zero_grad()
            output, target, inp = forward_pass(
                model, batch, gene_ids, include_zero_gene, device, amp, max_seq_len
            )
            mask = torch.ones_like(inp, dtype=torch.bool)
            loss = masked_mse_loss(output, target, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimiser)
            scaler.update()

            cells_seen += len(batch.y)
            steps += 1
            epoch_loss += float(loss.item())
            epoch_batches += 1

        metrics = evaluate_val(
            model,
            inputs.val_loader,
            inputs.ctrl_adata,
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
        wb.log(
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
        wall_clock_s=time.time() - t0,
        wandb_run_url=wb.url(),
        reason=reason,
        details={
            "num_epochs_trained": epochs_trained,
            "cells_seen": cells_seen,
            "steps": steps,
            "best_val_metrics": best_metrics,
            "best_val_epoch": best_epoch,
            "stop_metric": stop_metric,
        },
    )
    print(
        f"==> fine-tune done: reason={reason}, "
        f"best {stop_metric}={best_val_score:.4f} at epoch {best_epoch}, "
        f"cells_seen={cells_seen:,}, wall={stats.wall_clock_s/60:.1f}min"
    )
    return model, stats


def finetune_cache_dir(dataset: str, *, variant: str = "base") -> Path:
    """Per-(dataset, variant) cache directory under models/scgpt/checkpoints/.

    `variant` discriminates architectural changes that would otherwise
    silently reuse the wrong cache (e.g. base vs gene-prior-augmented).
    """
    return CKPT_ROOT / f"{dataset}_{variant}_ft"


def maybe_finetune(
    model: TransformerGenerator,
    inputs: ScgptInputs,
    gene_ids: np.ndarray,
    dataset: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[TransformerGenerator, TrainStats]:
    """Resolve the fine-tuned weights via cache_or_train, mutating model in place."""
    variant = "esm" if model.gene_prior is not None else "base"
    cache_dir = finetune_cache_dir(dataset, variant=variant)

    def _load(cache: Path) -> TransformerGenerator:
        load_finetuned_weights(model, cache / "best_model.pt", device)
        return model

    def _train() -> tuple[TransformerGenerator, TrainStats]:
        return finetune(
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

    def _save(m: TransformerGenerator, cache: Path) -> None:
        torch.save(m.state_dict(), cache / "best_model.pt")

    return cache_or_train(
        cache_dir=cache_dir,
        files=FT_FILES,
        hf_repo=args.hf_repo,
        force=args.force,
        load_cached=_load,
        train=_train,
        save_trained=_save,
    )


# ── Inference and saving ─────────────────────────────────────────────────────


@torch.no_grad()
def predict(
    model: TransformerGenerator,
    loader: Iterable,
    gene_ids: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
    limit_num_batches: int | None = None,
) -> dict:
    model.eval()
    pert_cat: list[str] = []
    cell_ids: list[str] = []
    preds: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []

    total = len(loader) if limit_num_batches is None else min(len(loader), limit_num_batches)
    it = loader if limit_num_batches is None else islice(loader, limit_num_batches)
    for batch in tqdm(it, total=total, desc="scgpt predict"):
        batch.to(device)
        pert_cat.extend(batch.pert)
        cell_ids.extend(batch.obs_name)
        p = model.pred_perturb(
            batch, include_zero_gene=include_zero_gene, gene_ids=gene_ids
        )
        preds.extend(p.cpu())
        truths.extend(batch.y.cpu())

    return {
        "pert": np.array(pert_cat),
        "cell_id": np.array(cell_ids, dtype=object),
        "pred": torch.stack(preds).numpy().astype(np.float32),
        "truth": torch.stack(truths).numpy().astype(np.float32),
    }


@torch.no_grad()
def predict_with_capture(
    model: TransformerGenerator,
    loader: Iterable,
    gene_ids: np.ndarray,
    gene_symbols: np.ndarray,
    include_zero_gene: str,
    device: torch.device,
    max_seq_len: int,
    activation_out: Path,
    capture_dtype: torch.dtype,
    dataset: str,
    split: str,
    batches_per_shard: int | None = None,
    limit_num_batches: int | None = None,
    capture_layers: str = "all",
) -> dict:
    """predict() variant that captures per-layer hidden states via HookManager.

    Mirrors pred_perturb's input prep + scatter-back, but the forward goes
    through nnsight so transformer_encoder.layers[i].output is captured per
    batch into an HDF5 sink. Capture path runs fp32 (no autocast) for stable
    downstream probe/SAE/reconstruction tolerances.
    """
    model.eval()
    # Forward order per _encode: encoder → value_encoder → pert_encoder (these
    # three sum into the residual stream entry point) → transformer_encoder
    # layers. Layout BTD because TransformerEncoderLayer uses batch_first=True.
    targets = [
        ("encoder",        lambda m: m.encoder.output,        "BTD"),
        ("value_encoder",  lambda m: m.value_encoder.output,  "BTD"),
        ("pert_encoder",   lambda m: m.pert_encoder.output,   "BTD"),
        *[(f"transformer_encoder.layers.{i}",
           lambda m, i=i: m.transformer_encoder.layers[i].output,
           "BTD")
          for i in range(len(model.transformer_encoder.layers))],
    ]
    if capture_layers != "all":
        requested = {s.strip() for s in capture_layers.split(",") if s.strip()}
        valid = {name for name, *_ in targets}
        unknown = sorted(requested - valid)
        if unknown:
            raise ValueError(
                f"unknown capture layer(s) {unknown}; valid: {sorted(valid)}"
            )
        targets = [t for t in targets if t[0] in requested]
        print(f"==> capturing {len(targets)} layer(s): {[t[0] for t in targets]}")
    # Force the dense forward path on the transformer encoder. With
    # src_key_padding_mask set and no flash_attn available, nn.TransformerEncoder
    # packs the batch into a NestedTensor between layers and only unpacks at
    # encoder exit — so per-layer `.output` proxies resolve to NestedTensors,
    # which have no usable .shape and cannot be written to the h5 sink. The
    # dense path is numerically equivalent at valid token positions (PyTorch's
    # documented guarantee), so predictions are unchanged. `use_nested_tensor`
    # is the runtime-read flag (derived from `enable_nested_tensor` at init);
    # flipping the ctor arg after-the-fact does nothing, so set this one.
    model.transformer_encoder.use_nested_tensor = False
    nn_model = NNsight(model)

    pert_cat: list[str] = []
    cell_ids: list[str] = []
    preds: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []

    sink = H5ActivationSink(
        activation_out,
        runner="scgpt",
        dataset=dataset,
        split=split,
        capture_names=[name for name, *_ in targets],
        # Run-level constants that make the h5 self-describing. gene_symbols
        # is ordered by dataset column; resolve a token's gene via
        # gene_symbols[gene_dataset_ids[cell, token]]. include_zero_gene
        # documents whether gene_dataset_ids is a sampled window or the full
        # gene set.
        extra_meta={
            "gene_symbols": gene_symbols,
            "include_zero_gene": include_zero_gene,
        },
        batches_per_shard=batches_per_shard,
    )
    with sink, HookManager(
        nn_model, capture=targets, sink=sink, capture_dtype=capture_dtype
    ) as hm:
        hm.set_tag("phase", "predict")
        cell_offset = 0
        total = len(loader) if limit_num_batches is None else min(len(loader), limit_num_batches)
        it = loader if limit_num_batches is None else islice(loader, limit_num_batches)
        for batch in tqdm(it, total=total, desc="scgpt capture"):
            pert_cat.extend(batch.pert)
            fa = build_forward_args(
                batch, gene_ids, include_zero_gene, device, max_seq_len
            )
            bs = fa.input_values.shape[0]
            # Gene dataset indices are sampled/permuted per batch (see
            # build_forward_args), so they're stored per-cell even though all
            # cells in one batch share the same window. .expand is a view;
            # the sink materializes to numpy, so no mem blowup before then.
            hm.set_per_cell({
                "cell_id": np.array(batch.obs_name, dtype=object),
                "cell_index": torch.arange(cell_offset, cell_offset + bs),
                "pert": np.array(batch.pert, dtype=object),
                "gene_dataset_ids": fa.input_gene_ids.unsqueeze(0).expand(bs, -1).contiguous().cpu(),
            })
            output_dict = hm.run(
                fa.mapped_input_gene_ids,
                fa.input_values,
                fa.input_pert_flags,
                src_key_padding_mask=fa.src_key_padding_mask,
                CLS=False,
                CCE=False,
                MVC=False,
                ECS=False,
                do_sample=True,
            )
            output_values = output_dict["mlm_output"].float()
            pred_full = scatter_back(output_values, fa.input_gene_ids, fa.n_genes)
            preds.extend(pred_full.cpu())
            truths.extend(batch.y.cpu())
            cell_ids.extend(batch.obs_name)
            cell_offset += bs

    return {
        "pert": np.array(pert_cat),
        "cell_id": np.array(cell_ids, dtype=object),
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

    # Sample mode already balance-samples controls into the loader output
    # via the (cell_line, non-targeting) buckets, so re-appending the full
    # ctrl_adata pool would duplicate those cells in the h5ad. Skip the
    # append for sample; keep it for train/val/test where it gives Cell-Eval
    # the full basal reference distribution.
    if split == "sample":
        pred_X = results["pred"]
        truth_X = results["truth"]
        labels = results["pert"]
        cell_ids = results["cell_id"]
        n_ctrl_appended = 0
    else:
        ctrl_X = ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, "toarray") else np.asarray(ctrl_adata.X)
        pred_X = np.vstack([results["pred"], ctrl_X])
        truth_X = np.vstack([results["truth"], ctrl_X])
        labels = np.concatenate(
            [results["pert"], np.array([control_label] * ctrl_adata.n_obs)]
        )
        cell_ids = np.concatenate(
            [results["cell_id"], ctrl_adata.obs.index.values.astype(object)]
        )
        n_ctrl_appended = ctrl_adata.n_obs

    obs = pd.DataFrame({pert_col: labels}, index=pd.Index(cell_ids, name="cell_id"))
    adata = ad.AnnData(
        X=pred_X,
        obs=obs,
        var=var.copy(),
        layers={"truth": truth_X},
    )
    adata.uns["model"] = "scgpt"
    adata.uns["dataset"] = dataset
    adata.uns["split"] = split
    adata.uns["pert_col"] = pert_col
    adata.uns["control_label"] = control_label
    adata.uns["train_stats"] = train_stats.__dict__
    adata.write_h5ad(output)
    print(f"==> wrote {adata.shape} to {output} ({n_ctrl_appended} control cells appended)")


# ── Runner spec ───────────────────────────────────────────────────────────────


@dataclass
class ScgptTrained:
    """State produced by train_or_load and consumed by predict."""
    model: TransformerGenerator
    gene_ids: np.ndarray
    device: torch.device


def _add_args(p: argparse.ArgumentParser) -> None:
    """Add scgpt-specific flags on top of the common ones."""
    p.add_argument(
        "--pretrained-dir",
        type=Path,
        default=DEFAULT_PRETRAINED,
        help="scGPT foundation checkpoint folder",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument(
        "--include-zero-gene", default="all", choices=["all", "batch-wise"]
    )
    p.add_argument("--split-type", default="simulation")
    p.add_argument("--num-epochs", type=int, default=15)
    p.add_argument("--early-stop", type=int, default=10)
    p.add_argument(
        "--stop-metric",
        default="pearson",
        choices=["pearson", "pearson_delta", "pearson_de", "pearson_de_delta"],
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--max-seq-len", type=int, default=1536)
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--limit-num-batches", type=int, default=None)
    p.add_argument(
        "--gene-prior-path",
        type=Path,
        default=None,
        help="safetensors with [vocab, prior_dim] frozen per-gene prior table",
    )


def _train_or_load(
    inputs: ScgptInputs, dataset: str, args: argparse.Namespace
) -> tuple[ScgptTrained, TrainStats]:
    """Build the transformer from pretrained weights, then finetune or load cache."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> device: {device}")

    vocab = build_vocab(args.pretrained_dir / "vocab.json")
    with open(args.pretrained_dir / "args.json") as f:
        margs = json.load(f)
    gene_ids = build_gene_ids(inputs.var, vocab)
    gene_prior = None
    if args.gene_prior_path is not None:
        gene_prior = GenePriorEncoder.from_safetensors(
            args.gene_prior_path, d_model=margs["embsize"]
        )
        print(f"==> loaded gene prior from {args.gene_prior_path}")
    model = build_model(margs, vocab, gene_prior=gene_prior).to(device)
    load_pretrained_weights(model, args.pretrained_dir / "best_model.pt", device)

    if args.skip_finetune:
        stats = TrainStats(
            wall_clock_s=0.0,
            wandb_run_url=None,
            reason="skipped",
            details={"mode": "skip_finetune", "pretrained_dir": str(args.pretrained_dir)},
        )
        return ScgptTrained(model=model, gene_ids=gene_ids, device=device), stats

    model, stats = maybe_finetune(model, inputs, gene_ids, dataset, device, args)
    return ScgptTrained(model=model, gene_ids=gene_ids, device=device), stats


def _predict(
    trained: ScgptTrained, inputs: ScgptInputs, args: argparse.Namespace
) -> dict:
    """Run pred_perturb over the requested split and return the results dict.

    With --capture-activations, swaps the inner forward for an nnsight-traced
    one that streams per-layer hidden states to args.activation_out.
    """
    loader = {
        "train": inputs.train_loader,
        "val": inputs.val_loader,
        "test": inputs.test_loader,
        "sample": inputs.sample_loader,
    }[args.split]
    if loader is None:
        raise RuntimeError(
            f"--split {args.split!r} requested but the loader is None; "
            f"the source ({type(inputs).__name__}) doesn't expose this split"
        )
    print(f"==> running inference on {args.split} split")
    if not args.capture_activations:
        return predict(
            trained.model, loader, trained.gene_ids, args.include_zero_gene, trained.device,
            limit_num_batches=args.limit_num_batches,
        )
    activation_out = args.activation_out or default_activation_out(
        REPO_ROOT, "scgpt", args.dataset, args.split
    )
    capture_dtype = {"fp32": torch.float32, "fp16": torch.float16}[args.capture_dtype]
    print(f"==> capturing activations to {activation_out} (dtype={args.capture_dtype})")
    # gene_symbols[i] names dataset column i; stored once in /meta so
    # downstream can resolve the per-cell gene_dataset_ids labels back to
    # gene names without joining against the prediction h5ad.
    gene_symbols = inputs.var["gene_name"].to_numpy().astype(object)
    return predict_with_capture(
        trained.model,
        loader,
        trained.gene_ids,
        gene_symbols,
        args.include_zero_gene,
        trained.device,
        args.max_seq_len,
        activation_out,
        capture_dtype,
        args.dataset,
        args.split,
        batches_per_shard=args.batches_per_shard,
        limit_num_batches=args.limit_num_batches,
        capture_layers=args.capture_layers,
    )


def _save_predictions(
    results: dict,
    inputs: ScgptInputs,
    manifest: Manifest,
    args: argparse.Namespace,
    stats: TrainStats,
    output: Path,
) -> None:
    """Invoke the module-level save_predictions with resolved manifest fields."""
    save_predictions(
        results,
        inputs.var,
        output,
        args.dataset,
        args.split,
        stats,
        inputs.ctrl_adata,
        manifest.obs.pert_col,
        manifest.obs.control_label,
    )


SPEC = RunnerSpec(
    name="scgpt",
    add_args=_add_args,
    load_inputs=load_dataset,
    train_or_load=_train_or_load,
    predict=_predict,
    save_predictions=_save_predictions,
)


if __name__ == "__main__":
    from scripts.runner import run
    run(SPEC)
