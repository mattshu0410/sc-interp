"""
Train CellFlow on a perturbation dataset and write predictions.

Usage:
    source models/cellflow/.venv/bin/activate
    python -m scripts.run cellflow --dataset norman --split test

If a cached trained model exists at models/cellflow/checkpoints/<dataset>/,
training is skipped. --force to retrain.

CellFlow consumes AnnData with gene perturbations encoded as two obs
columns (gene1, gene2) plus an ESM2 embedding dict in adata.uns. We
generate those from GEARS's processed AnnData on first run and reuse
GEARS's simulation split so the eval set matches the scGPT runner.
"""

import argparse
import functools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import anndata as ad
import numpy as np
import pandas as pd

import cellflow
import cellflow.preprocessing as cfpp
import cellflow.training as cftrain
import optax
from cellflow.model import CellFlow
from cellflow.preprocessing import get_esm_embedding
from ott.solvers import utils as solver_utils

from scripts import wb
from scripts.cache import TrainStats, cache_or_train
from scripts.data.genes import build_symbol_to_id
from scripts.data.splits import load_split
from scripts.manifest import Manifest
from scripts.runner import RunnerSpec

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO_ROOT / "models" / "cellflow" / "checkpoints"
CF_FILES = ["CellFlow.pkl"]

# Control label used on CellFlow's side. The condition parser translates
# GEARS's "ctrl" labels into this value on both gene slots.
CF_CONTROL = "control"


# ── Dataset dispatch ──────────────────────────────────────────────────────────
# Registry of manifest-source handlers. Same pattern as run_scgpt.py:
# add a new source by registering a handler that returns CellFlowInputs.


@dataclass
class CellFlowInputs:
    adata_train: ad.AnnData      # train perturbations only (no val, no test)
    adata_val: ad.AnnData        # val perturbations + control cells
    adata_test: ad.AnnData       # test perturbations + control cells
    ctrl_adata: ad.AnnData       # real control cells, shared baseline
    var: pd.DataFrame            # gene metadata
    test_conditions: list[str]   # held-out test perturbation labels from GEARS


def _parse_condition(
    cond: str, symbol_to_id: dict[str, str]
) -> tuple[str, str]:
    """Map GEARS condition labels to a (gene1, gene2) pair of Ensembl IDs.

    GEARS condition labels use gene symbols ('TGFBR2+ETS2'). CellFlow's
    get_esm_embedding expects Ensembl gene IDs so it can query the REST
    API for the canonical transcript. We translate here using the
    symbol -> id dict built from adata.var.

    'ctrl'        -> ('control', 'control')
    'GENE+ctrl'   -> ('ENSG...', 'control')
    'ctrl+GENE'   -> ('control', 'ENSG...')
    'GENE1+GENE2' -> ('ENSG...', 'ENSG...')
    """
    if cond == "ctrl":
        return CF_CONTROL, CF_CONTROL
    a, b = cond.split("+")
    def _map(sym: str) -> str:
        if sym == "ctrl":
            return CF_CONTROL
        if sym not in symbol_to_id:
            raise KeyError(f"gene symbol {sym!r} not found in var gene_name lookup")
        return symbol_to_id[sym]
    return _map(a), _map(b)


def _load_gears(manifest: Manifest, args: argparse.Namespace) -> CellFlowInputs:
    """Load a GEARS-processed AnnData and read the canonical split JSON.

    Does not import gears. The processed h5ad and the canonical split JSON
    are produced by scripts.data.gears running in the tools venv.
    """
    split_cfg = manifest.raw.get("split", {}) or {}
    split_type = split_cfg.get("split_type", args.split_type)
    seed = split_cfg.get("seed", args.seed)
    train_gene_set_size = split_cfg.get("train_gene_set_size", 0.75)

    gears_name = manifest.raw["gears_name"]
    dataset_dir = REPO_ROOT / "data" / gears_name
    adata = ad.read_h5ad(dataset_dir / "perturb_processed.h5ad")

    split = load_split(manifest, split_type, seed, train_gene_set_size)

    pert_col = manifest.obs.pert_col
    ctrl_label = manifest.obs.control_label

    if manifest.var is None or manifest.var.gene_id_type != "ensembl":
        raise ValueError(
            f"cellflow gears loader requires manifest.var with "
            f"gene_id_type=ensembl for ESM lookup, got "
            f"{manifest.var.gene_id_type if manifest.var else None!r}"
        )
    symbol_to_id = build_symbol_to_id(adata, manifest.var)

    # Column names match the CellFlow reproducibility repo's Norman config
    # (gene_1/gene_2, control, esm2) so this adata is a drop-in fit for
    # their prepare_data call shape.
    genes = np.array(
        [_parse_condition(c, symbol_to_id) for c in adata.obs[pert_col].astype(str)],
        dtype=object,
    )
    adata.obs["gene_1"] = genes[:, 0]
    adata.obs["gene_2"] = genes[:, 1]
    adata.obs["control"] = (adata.obs[pert_col] == ctrl_label).values
    # Keep "condition" as the human-readable label used as condition_id_key
    adata.obs["condition"] = adata.obs[pert_col].astype(str)

    # Densify X so CellFlow's JAX path and centered_pca can consume it
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray().astype(np.float32)

    # ESM2 embeddings for every unique gene across both slots. Stored
    # under 'esm2' in adata.uns to match the paper's conf/dataset/norman.yaml.
    print("==> generating ESM2 embeddings for unique perturbed genes")
    get_esm_embedding(
        adata,
        gene_key=["gene_1", "gene_2"],
        null_value=CF_CONTROL,
        gene_emb_key="esm2",
        esm_model_name=args.esm_model,
    )
    for k, v in list(adata.uns["esm2"].items()):
        if hasattr(v, "detach"):
            adata.uns["esm2"][k] = v.detach().cpu().numpy()
    emb_dim = next(iter(adata.uns["esm2"].values())).shape[0]
    adata.uns["esm2"][CF_CONTROL] = np.zeros(emb_dim, dtype=np.float32)

    # Three-way split from GEARS simulation. Controls are a shared baseline
    # and get included in all three AnnDatas so CellFlow can do
    # control->pert transport during both training and evaluation.
    train_perts = set(split["train"])
    val_perts = set(split["val"])
    test_perts = set(split["test"])
    is_ctrl = adata.obs["control"].values

    adata_train = adata[adata.obs[pert_col].isin(train_perts).values | is_ctrl].copy()
    adata_val = adata[adata.obs[pert_col].isin(val_perts).values | is_ctrl].copy()
    adata_test = adata[adata.obs[pert_col].isin(test_perts).values | is_ctrl].copy()
    ctrl_adata = adata[is_ctrl].copy()

    # 50-dim PCA on the training cells, then project val and test onto it.
    # Matches the paper's X_pca representation used as sample_rep.
    print("==> fitting 50-dim PCA on training cells")
    cfpp.centered_pca(adata_train, n_comps=50, keep_centered_data=False)
    cfpp.project_pca(adata_val, ref_adata=adata_train)
    cfpp.project_pca(adata_test, ref_adata=adata_train)

    print(
        f"==> train cells: {adata_train.n_obs}, val cells: {adata_val.n_obs}, "
        f"test cells: {adata_test.n_obs}, controls: {ctrl_adata.n_obs}"
    )
    print(
        f"==> train perts: {len(train_perts)}, val perts: {len(val_perts)}, "
        f"test perts: {len(test_perts)}"
    )

    return CellFlowInputs(
        adata_train=adata_train,
        adata_val=adata_val,
        adata_test=adata_test,
        ctrl_adata=ctrl_adata,
        var=adata.var.copy(),
        test_conditions=sorted(test_perts),
    )


LOADERS: dict[str, Callable[[Manifest, argparse.Namespace], CellFlowInputs]] = {
    "gears": _load_gears,
}


def load_dataset(manifest: Manifest, args: argparse.Namespace) -> CellFlowInputs:
    """Dispatch to the LOADERS handler for manifest.source."""
    if manifest.source not in LOADERS:
        raise NotImplementedError(
            f"manifest source {manifest.source!r} not supported by run_cellflow. "
            f"registered: {sorted(LOADERS)}"
        )
    return LOADERS[manifest.source](manifest, args)


# ── Training ──────────────────────────────────────────────────────────────────


def train_cellflow(
    inputs: CellFlowInputs, args: argparse.Namespace, wandb_config: dict
) -> tuple[CellFlow, TrainStats]:
    """Paper-faithful training of CellFlow on Norman.

    All hyperparameters, architecture, and callback setup mirror the
    reproducibility repo's conf/model/norman.yaml and train_norman_pca_50.py.
    The only change is our GEARS simulation split vs the paper's biolord
    splits, which is deliberate so our three-way comparison with scGPT and
    scLDM uses a single split definition.
    """
    cf = CellFlow(inputs.adata_train, solver="otfm")

    cf.prepare_data(
        sample_rep="X_pca",
        control_key="control",
        perturbation_covariates={"target_gene": ("gene_1", "gene_2")},
        perturbation_covariate_reps={"target_gene": "esm2"},
        sample_covariates=None,
        sample_covariate_reps=None,
        split_covariates=None,
    )

    cf.prepare_validation_data(
        inputs.adata_val,
        name="val",
        n_conditions_on_log_iteration=5,
        n_conditions_on_train_end=5,
    )

    match_fn = functools.partial(
        solver_utils.match_linear,
        epsilon=0.1,
        scale_cost="mean",
        tau_a=1.0,
        tau_b=1.0,
    )
    optimizer = optax.MultiSteps(optax.adam(5e-5), 20)
    flow = {"constant_noise": 1.0}

    cf.prepare_model(
        condition_embedding_dim=1024,
        pooling="attention_token",
        time_encoder_dims=(2048, 2048, 2048),
        time_encoder_dropout=0.0,
        hidden_dims=(4096, 4096, 4096),
        hidden_dropout=0.0,
        decoder_dims=(4096, 4096, 4096),
        decoder_dropout=0.2,
        layers_before_pool={
            "target_gene": {
                "layer_type": "mlp",
                "dims": [1024, 1024],
                "dropout_rate": 0.5,
            }
        },
        layers_after_pool={
            "layer_type": "mlp",
            "dims": [1024, 1024],
            "dropout_rate": 0.2,
        },
        cond_output_dropout=0.9,
        time_freqs=1024,
        match_fn=match_fn,
        optimizer=optimizer,
        probability_path=flow,
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
    )

    # Callbacks match the reproducibility repo: Metrics + PCADecodedMetrics
    # on the val set, plus WandbLogger which calls wandb.init() internally.
    metrics_cb = cftrain.Metrics(metrics=["r_squared", "mmd", "e_distance"])
    decoded_cb = cftrain.PCADecodedMetrics(
        ref_adata=inputs.adata_train,
        metrics=["r_squared", "mmd", "e_distance"],
    )
    wandb_cb = cftrain.WandbLogger(
        project=args.wandb_project,
        out_dir=str(CKPT_ROOT),
        config=wandb_config,
    )
    callbacks = [metrics_cb, decoded_cb, wandb_cb]

    print(
        f"==> training CellFlow for {args.num_iterations:,} iterations "
        f"(batch {args.batch_size}, valid_freq {args.valid_freq})"
    )
    t0 = time.time()
    cf.train(
        num_iterations=args.num_iterations,
        batch_size=args.batch_size,
        callbacks=callbacks,
        valid_freq=args.valid_freq,
    )
    wall = time.time() - t0

    # WandbLogger initialised a run, grab its URL for TrainStats
    try:
        import wandb
        run_url = wandb.run.url if wandb.run is not None else None
    except Exception:
        run_url = None

    stats = TrainStats(
        wall_clock_s=wall,
        wandb_run_url=run_url,
        details={
            "iterations": args.num_iterations,
            "batch_size": args.batch_size,
            "valid_freq": args.valid_freq,
        },
    )
    print(f"==> training done in {wall/60:.1f} min")
    return cf, stats


def maybe_train(
    inputs: CellFlowInputs, dataset: str, args: argparse.Namespace
) -> tuple[CellFlow, TrainStats]:
    """Resolve trained CellFlow weights via cache_or_train."""
    cache_dir = CKPT_ROOT / dataset
    wandb_config = {
        "model": "cellflow",
        "dataset": dataset,
        "num_iterations": args.num_iterations,
        "batch_size": args.batch_size,
        "valid_freq": args.valid_freq,
        "sample_rep": "X_pca",
        "pca_dim": 50,
        "learning_rate": 5e-5,
        "multi_steps": 20,
        "split_type": "gears_simulation",
        "seed": args.seed,
    }

    def _load(cache: Path) -> CellFlow:
        return CellFlow.load(str(cache / "CellFlow.pkl"))

    def _train() -> tuple[CellFlow, TrainStats]:
        return train_cellflow(inputs, args, wandb_config)

    def _save(cf: CellFlow, cache: Path) -> None:
        cf.save(str(cache), overwrite=True)

    return cache_or_train(
        cache_dir=cache_dir,
        files=CF_FILES,
        hf_repo=args.hf_repo,
        force=args.force,
        load_cached=_load,
        train=_train,
        save_trained=_save,
    )


# ── Inference and saving ─────────────────────────────────────────────────────


def predict(cf: CellFlow, inputs: CellFlowInputs) -> dict:
    """Run cf.predict on held-out test perturbations in PCA space, then
    invert the PCA back to gene space.

    Returns a dict {condition -> (n_cells, n_genes)} of predicted expression
    in the original gene space, ready for cell-eval.
    """
    adata_ctrl = inputs.adata_test[inputs.adata_test.obs["control"].values]
    test_obs = inputs.adata_test[~inputs.adata_test.obs["control"].values].obs
    covariate_data = test_obs.drop_duplicates(subset=["gene_1", "gene_2"]).copy()

    print(f"==> predicting {len(covariate_data)} test perturbations (X_pca)")
    preds_pca = cf.predict(
        adata=adata_ctrl,
        sample_rep="X_pca",
        condition_id_key="condition",
        covariate_data=covariate_data,
    )

    # Invert PCA back to gene space via cfpp.reconstruct_pca. The function
    # wants an AnnData with the PCA'd predictions in obsm, and writes the
    # reconstructed gene-space values to adata.layers[layers_key_added].
    preds_gene: dict[str, np.ndarray] = {}
    for cond, arr in preds_pca.items():
        arr = np.asarray(np.squeeze(arr))
        tmp_adata = ad.AnnData(
            X=np.empty((arr.shape[0], inputs.adata_train.n_vars), dtype=np.float32),
            obs=pd.DataFrame({"condition": [cond] * arr.shape[0]}),
            var=inputs.adata_train.var.copy(),
        )
        tmp_adata.obsm["X_pca_pred"] = arr
        cfpp.reconstruct_pca(
            query_adata=tmp_adata,
            use_rep="X_pca_pred",
            ref_adata=inputs.adata_train,
            layers_key_added="X_recon",
        )
        preds_gene[cond] = np.asarray(tmp_adata.layers["X_recon"])
    return preds_gene


def save_predictions(
    preds: dict,
    inputs: CellFlowInputs,
    output: Path,
    dataset: str,
    split: str,
    train_stats: TrainStats,
    pert_col: str,
    control_label: str,
) -> None:
    """Write canonical prediction h5ad: pred in X, truth in layers['truth'],
    including control cells with the same obs schema as run_scgpt.py.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    # Pull real perturbed cells to use as the truth layer, matched per condition
    test_adata = inputs.adata_test[~inputs.adata_test.obs["control"].values]
    real_by_cond: dict[str, np.ndarray] = {}
    for cond in preds:
        mask = (test_adata.obs["condition"] == cond).values
        real_by_cond[cond] = (
            test_adata.X[mask].toarray() if hasattr(test_adata.X, "toarray") else test_adata.X[mask]
        )

    pred_rows = []
    truth_rows = []
    labels = []
    for cond, arr in preds.items():
        arr = np.asarray(np.squeeze(arr))
        real = real_by_cond[cond]
        # If pred and real cell counts differ, trim both to the shorter one so
        # row-aligned layers stay consistent. cell-eval computes distributional
        # metrics so exact per-cell alignment isn't required, it just wants both
        # to have the same shape.
        n = min(arr.shape[0], real.shape[0])
        pred_rows.append(arr[:n])
        truth_rows.append(real[:n])
        labels.extend([cond] * n)

    pred_X = np.vstack(pred_rows).astype(np.float32)
    truth_X = np.vstack(truth_rows).astype(np.float32)

    # Inject real controls on both X and truth so cell-eval can compute deltas
    ctrl_X = (
        inputs.ctrl_adata.X.toarray()
        if hasattr(inputs.ctrl_adata.X, "toarray")
        else np.asarray(inputs.ctrl_adata.X)
    ).astype(np.float32)
    pred_X = np.vstack([pred_X, ctrl_X])
    truth_X = np.vstack([truth_X, ctrl_X])
    labels.extend([control_label] * inputs.ctrl_adata.n_obs)

    adata_out = ad.AnnData(
        X=pred_X,
        obs=pd.DataFrame({pert_col: labels}),
        var=inputs.var.copy(),
        layers={"truth": truth_X},
    )
    adata_out.uns["model"] = "cellflow"
    adata_out.uns["dataset"] = dataset
    adata_out.uns["split"] = split
    adata_out.uns["pert_col"] = pert_col
    adata_out.uns["control_label"] = control_label
    adata_out.uns["train_stats"] = train_stats.__dict__
    adata_out.write_h5ad(output)
    print(f"==> wrote {adata_out.shape} to {output}")


# ── Runner spec ───────────────────────────────────────────────────────────────


def _add_args(p: argparse.ArgumentParser) -> None:
    """Add cellflow-specific flags on top of the common ones."""
    p.add_argument("--split-type", default="simulation")
    p.add_argument(
        "--esm-model",
        default="esm2_t6_8M_UR50D",
        help="ESM2 model for gene embeddings",
    )
    p.add_argument("--num-iterations", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--valid-freq", type=int, default=400_000)


def _train_or_load(
    inputs: CellFlowInputs, dataset: str, args: argparse.Namespace
) -> tuple[CellFlow, TrainStats]:
    """Resolve the trained CellFlow via maybe_train."""
    return maybe_train(inputs, dataset, args)


def _predict(
    cf: CellFlow, inputs: CellFlowInputs, args: argparse.Namespace
) -> dict:
    """Run cf.predict on held-out test perturbations."""
    return predict(cf, inputs)


def _save_predictions(
    preds: dict,
    inputs: CellFlowInputs,
    manifest: Manifest,
    args: argparse.Namespace,
    stats: TrainStats,
    output: Path,
) -> None:
    """Invoke the module-level save_predictions with resolved manifest fields."""
    save_predictions(
        preds,
        inputs,
        output,
        args.dataset,
        args.split,
        stats,
        manifest.obs.pert_col,
        manifest.obs.control_label,
    )


SPEC = RunnerSpec(
    name="cellflow",
    add_args=_add_args,
    load_inputs=load_dataset,
    train_or_load=_train_or_load,
    predict=_predict,
    save_predictions=_save_predictions,
)


if __name__ == "__main__":
    from scripts.runner import run
    run(SPEC)
