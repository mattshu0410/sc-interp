"""Train GEARS on a perturbation dataset and write predictions.

Usage:
    source models/gears/.venv/bin/activate
    python -m scripts.run gears --dataset norman --split test

GEARS has no foundation checkpoint, so this runner always trains from
scratch on first invocation. The result is cached at
`models/gears/checkpoints/<dataset>/` via cache_or_train; subsequent
invocations load the cache. --force to retrain.

The paper's training recipe (Roohani et al. 2024, fig2_train.py in
GEARS_misc) is the default: 15 epochs, lr 1e-3, batch 32, hidden 64,
`simulation` split. GEARS needs `default_pert_graph=False` for Norman
so it builds the perturbation graph from the dataset's own genes.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import anndata as ad
import numpy as np
import pandas as pd
import torch
from nnsight import NNsight

from gears import GEARS, PertData

from scripts import wb
from scripts.cache import TrainStats, cache_or_train
from scripts.interp.hook_sinks import H5ActivationSink, default_activation_out
from scripts.interp.hooks import HookManager
from scripts.manifest import Manifest
from scripts.runner import RunnerSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO_ROOT / "models" / "gears" / "checkpoints"
GEARS_FILES = ["config.pkl", "model.pt"]


# ── Dataset dispatch ──────────────────────────────────────────────────────────


@dataclass
class GearsInputs:
    """Everything train/predict need from the dataset loader."""
    pert_data: PertData
    var: pd.DataFrame


def _load_gears(manifest: Manifest, args: argparse.Namespace) -> GearsInputs:
    """Build PertData + dataloaders for a manifest whose source is 'gears'."""
    pert_data = PertData(str(REPO_ROOT / "data"), default_pert_graph=False)
    pert_data.load(data_name=manifest.raw["gears_name"])
    pert_data.prepare_split(
        split=manifest.raw.get("split", {}).get("default", args.split_type),
        seed=args.seed,
        train_gene_set_size=0.75,
    )
    pert_data.get_dataloader(
        batch_size=args.batch_size, test_batch_size=args.eval_batch_size
    )
    return GearsInputs(pert_data=pert_data, var=pert_data.adata.var.copy())


LOADERS: dict[str, Callable[[Manifest, argparse.Namespace], GearsInputs]] = {
    "gears": _load_gears,
}


def load_dataset(manifest: Manifest, args: argparse.Namespace) -> GearsInputs:
    """Dispatch to the LOADERS handler for manifest.source."""
    if manifest.source not in LOADERS:
        raise NotImplementedError(
            f"manifest source {manifest.source!r} not supported by run_gears. "
            f"registered: {sorted(LOADERS)}"
        )
    return LOADERS[manifest.source](manifest, args)


# ── Training ──────────────────────────────────────────────────────────────────


def maybe_train(
    inputs: GearsInputs, dataset: str, args: argparse.Namespace
) -> tuple[GEARS, TrainStats]:
    """Resolve trained GEARS weights via cache_or_train.

    Cached artifacts live at models/gears/checkpoints/<dataset>/ and are
    what GEARS.save_model writes (config.pkl + model.pt).
    """
    cache_dir = CKPT_ROOT / dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"==> device: {device}")

    def _load(cache: Path) -> GEARS:
        model = GEARS(inputs.pert_data, device=device)
        model.load_pretrained(str(cache))
        return model

    def _train() -> tuple[GEARS, TrainStats]:
        model = GEARS(inputs.pert_data, device=device)
        model.model_initialize(
            hidden_size=args.hidden_size,
            num_go_gnn_layers=args.num_go_gnn_layers,
            num_gene_gnn_layers=args.num_gene_gnn_layers,
            num_similar_genes_go_graph=args.num_similar_genes_go,
            num_similar_genes_co_express_graph=args.num_similar_genes_coexpress,
            coexpress_threshold=args.coexpress_threshold,
            direction_lambda=args.direction_lambda,
            no_perturb=False,
        )
        print(
            f"==> training GEARS for {args.num_epochs} epochs "
            f"(hidden_size={args.hidden_size}, batch={args.batch_size}, lr={args.lr})"
        )
        t0 = time.time()
        model.train(
            epochs=args.num_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        wall = time.time() - t0
        stats = TrainStats(
            wall_clock_s=wall,
            wandb_run_url=wb.url(),
            reason="max_epochs",
            details={
                "num_epochs": args.num_epochs,
                "hidden_size": args.hidden_size,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "weight_decay": args.weight_decay,
                "coexpress_threshold": args.coexpress_threshold,
                "direction_lambda": args.direction_lambda,
            },
        )
        print(f"==> training done in {wall / 60:.1f} min")
        return model, stats

    def _save(model: GEARS, cache: Path) -> None:
        cache.mkdir(parents=True, exist_ok=True)
        model.save_model(str(cache))

    return cache_or_train(
        cache_dir=cache_dir,
        files=GEARS_FILES,
        hf_repo=args.hf_repo,
        force=args.force,
        load_cached=_load,
        train=_train,
        save_trained=_save,
    )


# ── Inference and saving ─────────────────────────────────────────────────────


@torch.no_grad()
def predict(model: GEARS, loader: Iterable, device: torch.device) -> dict:
    """Per-cell forward pass over the test loader.

    Iterates pert_data.dataloader['test_loader'] batch-by-batch, calls
    model.best_model on each batch to get per-cell predictions, and pairs
    them with the batch's ground-truth targets and perturbation labels.
    Matches the iteration pattern in gears.inference.evaluate.
    """
    model.best_model.eval()
    model.best_model.to(device)
    pert_cat: list[str] = []
    preds: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []

    for batch in loader:
        batch.to(device)
        pert_cat.extend(batch.pert)
        p = model.best_model(batch)
        preds.extend(p.cpu())
        truths.extend(batch.y.cpu())

    return {
        "pert": np.array(pert_cat),
        "pred": torch.stack(preds).numpy().astype(np.float32),
        "truth": torch.stack(truths).numpy().astype(np.float32),
    }


@torch.no_grad()
def predict_with_capture(
    model: GEARS,
    loader: Iterable,
    device: torch.device,
    activation_out: Path,
    capture_dtype: torch.dtype,
    dataset: str,
    split: str,
    gene_symbols: np.ndarray,
    batches_per_shard: int | None = None,
) -> dict:
    """predict() variant that captures GEARS submodule outputs via HookManager.

    GEARS flattens the (batch, gene) grid to (batch*genes, hidden) in its
    forward, so gene-axis targets are reshaped back to (N_cells, num_genes,
    hidden) inside the accessor — that way axis 0 of the stored tensor
    matches cell_id / pert labels and downstream code doesn't need to know
    num_genes to slice per cell.

    pert_emb and sim_layers.* are intentionally omitted: they operate on
    (num_perts, hidden), not (num_cells, …), so per-cell labels don't apply
    to them. Capture those separately if/when needed with pert-indexed
    sidecars. pert_fuse is likewise excluded (forward:166 empty-pert_index
    guard skips it on all-control batches, which would trip a mandatory
    .save() at trace exit).
    """
    best = model.best_model
    best.eval()
    best.to(device)

    n_genes = best.num_genes
    h = best.gene_emb.embedding_dim

    # Forward order per GEARS_Model.forward; nnsight 0.5 raises
    # OutOfOrderError if .save() calls aren't in execution order. Every
    # gene-axis target is reshaped inside the accessor from (N*G, H) to
    # (N, G, H); cross_gene_state already emits (N, H) and needs no reshape.
    targets = [
        ("gene_emb",
         lambda m, g=n_genes, h=h: m.gene_emb.output.reshape(-1, g, h)),
        ("emb_pos",
         lambda m, g=n_genes, h=h: m.emb_pos.output.reshape(-1, g, h)),
        *[(f"layers_emb_pos.{i}",
           lambda m, i=i, g=n_genes, h=h: m.layers_emb_pos[i].output.reshape(-1, g, h))
          for i in range(len(best.layers_emb_pos))],
        ("emb_trans_v2",
         lambda m, g=n_genes, h=h: m.emb_trans_v2.output.reshape(-1, g, h)),
        ("recovery_w",
         lambda m, g=n_genes, h=h: m.recovery_w.output.reshape(-1, g, h)),
        ("cross_gene_state", lambda m: m.cross_gene_state.output),
    ]
    nn_model = NNsight(best)

    pert_cat: list[str] = []
    preds: list[torch.Tensor] = []
    truths: list[torch.Tensor] = []

    sink = H5ActivationSink(
        activation_out,
        runner="gears",
        dataset=dataset,
        split=split,
        capture_names=[name for name, *_ in targets],
        # gene_symbols[j] names axis-1 index j of every gene-axis target
        # (after reshape). num_genes lets readers sanity-check the reshape.
        extra_meta={
            "gene_symbols": gene_symbols,
            "num_genes": n_genes,
        },
        batches_per_shard=batches_per_shard,
    )
    with sink, HookManager(
        nn_model, capture=targets, sink=sink, capture_dtype=capture_dtype
    ) as hm:
        hm.set_tag("phase", "predict")
        cell_offset = 0
        for batch in loader:
            batch.to(device)
            pert_cat.extend(batch.pert)
            bs = batch.num_graphs
            hm.set_per_cell({
                "cell_id": torch.arange(cell_offset, cell_offset + bs),
                "pert": np.array(batch.pert, dtype=object),
            })
            p = hm.run(batch)
            preds.extend(p.cpu())
            truths.extend(batch.y.cpu())
            cell_offset += bs

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
    """Write pred + truth + real controls to one canonical h5ad.

    Layout matches run_scgpt.py::save_predictions: .X = predicted expression
    (perturbed cells) + real control expression, .layers['truth'] = ground
    truth expression (perturbed cells) + real control expression, obs[pert_col]
    = perturbation label per row with control_label for control rows.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    ctrl_X = (
        ctrl_adata.X.toarray() if hasattr(ctrl_adata.X, "toarray")
        else np.asarray(ctrl_adata.X)
    )
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
    adata.uns["model"] = "gears"
    adata.uns["dataset"] = dataset
    adata.uns["split"] = split
    adata.uns["pert_col"] = pert_col
    adata.uns["control_label"] = control_label
    adata.uns["train_stats"] = train_stats.__dict__
    adata.write_h5ad(output)
    print(
        f"==> wrote {adata.shape} to {output} "
        f"({ctrl_adata.n_obs} control cells included)"
    )


# ── Runner spec ───────────────────────────────────────────────────────────────


def _add_args(p: argparse.ArgumentParser) -> None:
    """Add gears-specific flags on top of the common ones."""
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--split-type", default="simulation")
    p.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        help="epochs for GEARS.train, default 15 (paper's fig2_train.py)",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-go-gnn-layers", type=int, default=1)
    p.add_argument("--num-gene-gnn-layers", type=int, default=1)
    p.add_argument(
        "--num-similar-genes-go",
        type=int,
        default=20,
        help="top-k similar perturbations in GO-derived graph",
    )
    p.add_argument(
        "--num-similar-genes-coexpress",
        type=int,
        default=20,
        help="top-k similar genes in co-expression graph",
    )
    p.add_argument("--coexpress-threshold", type=float, default=0.4)
    p.add_argument("--direction-lambda", type=float, default=1e-1)


def _train_or_load(
    inputs: GearsInputs, dataset: str, args: argparse.Namespace
) -> tuple[GEARS, TrainStats]:
    """Train or load GEARS via cache_or_train."""
    return maybe_train(inputs, dataset, args)


def _predict(
    model: GEARS, inputs: GearsInputs, args: argparse.Namespace
) -> dict:
    """Run per-cell inference on the requested split.

    With --capture-activations, swaps the inner forward for an nnsight-traced
    one that streams submodule outputs to args.activation_out.
    """
    loader = {
        "train": inputs.pert_data.dataloader["train_loader"],
        "val": inputs.pert_data.dataloader["val_loader"],
        "test": inputs.pert_data.dataloader["test_loader"],
    }[args.split]
    device = next(model.best_model.parameters()).device
    print(f"==> running inference on {args.split} split")
    if not args.capture_activations:
        return predict(model, loader, device)
    activation_out = args.activation_out or default_activation_out(
        REPO_ROOT, "gears", args.dataset, args.split
    )
    capture_dtype = {"fp32": torch.float32, "fp16": torch.float16}[args.capture_dtype]
    print(f"==> capturing activations to {activation_out} (dtype={args.capture_dtype})")
    # gene_symbols[j] names axis-1 index j of every gene-axis activation in
    # the h5, so the file is self-describing without joining the prediction
    # .h5ad.
    gene_symbols = inputs.var["gene_name"].to_numpy().astype(object)
    return predict_with_capture(
        model,
        loader,
        device,
        activation_out,
        capture_dtype,
        args.dataset,
        args.split,
        gene_symbols,
        batches_per_shard=args.batches_per_shard,
    )


def _save_predictions(
    results: dict,
    inputs: GearsInputs,
    manifest: Manifest,
    args: argparse.Namespace,
    stats: TrainStats,
    output: Path,
) -> None:
    """Invoke the module-level save_predictions with resolved manifest fields."""
    pert_col = manifest.obs.pert_col
    control_label = manifest.obs.control_label
    ctrl_adata = inputs.pert_data.adata[
        inputs.pert_data.adata.obs[pert_col] == control_label
    ]
    save_predictions(
        results,
        inputs.var,
        output,
        args.dataset,
        args.split,
        stats,
        ctrl_adata,
        pert_col,
        control_label,
    )


SPEC = RunnerSpec(
    name="gears",
    add_args=_add_args,
    load_inputs=load_dataset,
    train_or_load=_train_or_load,
    predict=_predict,
    save_predictions=_save_predictions,
)


if __name__ == "__main__":
    from scripts.runner import run
    run(SPEC)
