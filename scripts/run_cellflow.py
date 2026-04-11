"""
Train CellFlow on a perturbation dataset and write predictions.

Usage:
    source models/cellflow/.venv/bin/activate
    python scripts/run_cellflow.py \\
        --dataset norman \\
        --split test

If a cached trained model exists at models/cellflow/checkpoints/<dataset>/,
training is skipped. --force-train to retrain.

CellFlow consumes AnnData with gene perturbations encoded as two obs
columns (gene1, gene2) plus an ESM2 embedding dict in adata.uns. We
generate those from GEARS's processed AnnData on first run and reuse
GEARS's simulation split so the eval set matches the scGPT runner.
"""

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import pickle

import anndata as ad
import numpy as np
import pandas as pd
import yaml

import functools

import cellflow
import cellflow.preprocessing as cfpp
import cellflow.training as cftrain
import optax
from cellflow.model import CellFlow
from cellflow.preprocessing import get_esm_embedding
from ott.solvers import utils as solver_utils

sys.path.insert(0, str(Path(__file__).parent))
import _hf
import _wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT_ROOT = REPO_ROOT / "models" / "cellflow" / "checkpoints"
CF_FILES = ["CellFlow.pkl", "training_stats.json"]

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


def _load_gears(manifest: dict, args: argparse.Namespace) -> CellFlowInputs:
    """Load a GEARS-processed dataset directly from disk.

    Reads the processed h5ad and the split pickle that were produced by
    the tools venv running gears.PertData. Avoids importing gears here
    so the cellflow venv doesn't need it as a dep.
    """
    split_cfg = manifest.get("split", {})
    split_type = split_cfg.get("split_type", args.split_type)
    seed = split_cfg.get("seed", args.seed)
    train_gene_set_size = split_cfg.get("train_gene_set_size", 0.75)

    dataset_dir = REPO_ROOT / "data" / manifest["gears_name"]
    adata = ad.read_h5ad(dataset_dir / "perturb_processed.h5ad")

    split_pkl = (
        dataset_dir
        / "splits"
        / f"{manifest['gears_name']}_{split_type}_{seed}_{train_gene_set_size}.pkl"
    )
    if not split_pkl.exists():
        raise FileNotFoundError(
            f"split pickle missing at {split_pkl}. Run the scgpt runner once "
            "to materialise it via gears.PertData.prepare_split, or call "
            "gears from the tools venv."
        )
    with open(split_pkl, "rb") as f:
        set2conditions = pickle.load(f)

    pert_col = manifest["obs"]["pert_col"]
    ctrl_label = manifest["obs"]["control_label"]

    # Build gene symbol -> Ensembl ID lookup from var. In Norman, var_names
    # are Ensembl IDs and the gene_name column has symbols.
    symbol_to_id = dict(zip(adata.var["gene_name"].astype(str), adata.var_names.astype(str)))

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
    train_perts = set(set2conditions["train"])
    val_perts = set(set2conditions["val"])
    test_perts = set(set2conditions["test"])
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


LOADERS: dict[str, Callable[[dict, argparse.Namespace], CellFlowInputs]] = {
    "gears": _load_gears,
}


def load_dataset(manifest: dict, args: argparse.Namespace) -> CellFlowInputs:
    source = manifest["source"]
    if source not in LOADERS:
        raise NotImplementedError(
            f"manifest source {source!r} not supported by run_cellflow. "
            f"registered: {sorted(LOADERS)}"
        )
    return LOADERS[source](manifest, args)


def load_manifest(dataset: str) -> dict:
    manifest_path = REPO_ROOT / "data" / dataset / "manifest.yaml"
    with open(manifest_path) as f:
        return yaml.safe_load(f)


# ── Training ──────────────────────────────────────────────────────────────────


@dataclass
class TrainStats:
    iterations: int
    wall_clock_s: float
    wandb_run_url: str | None = None


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
        iterations=args.num_iterations,
        wall_clock_s=wall,
        wandb_run_url=run_url,
    )
    print(f"==> training done in {wall/60:.1f} min")
    return cf, stats


def maybe_train(
    inputs: CellFlowInputs, dataset: str, args: argparse.Namespace
) -> tuple[CellFlow, TrainStats]:
    cache_dir = CKPT_ROOT / dataset
    ckpt_path = cache_dir / "CellFlow.pkl"
    stats_path = cache_dir / "training_stats.json"

    if not ckpt_path.exists() and args.hf_repo and not args.force_train:
        _hf.try_download(args.hf_repo, cache_dir, CF_FILES)

    if ckpt_path.exists() and not args.force_train:
        print(f"==> cached CellFlow found at {ckpt_path}, loading")
        cf = CellFlow.load(str(ckpt_path))
        if stats_path.exists():
            with open(stats_path) as f:
                stats = TrainStats(**json.load(f))
        else:
            stats = TrainStats(iterations=0, wall_clock_s=0.0)
        return cf, stats

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
    cf, stats = train_cellflow(inputs, args, wandb_config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cf.save(str(cache_dir), overwrite=True)
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"==> saved CellFlow to {ckpt_path}")

    if args.hf_repo:
        _hf.try_upload(args.hf_repo, cache_dir, CF_FILES)

    return cf, stats


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
    adata_out.uns["train_stats"] = asdict(train_stats)
    adata_out.write_h5ad(output)
    print(f"==> wrote {adata_out.shape} to {output}")


# ── Entry point ───────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--output", type=Path)
    p.add_argument("--split-type", default="simulation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--esm-model",
        default="esm2_t6_8M_UR50D",
        help="ESM2 model for gene embeddings, default the 8M param version",
    )

    # Training knobs (defaults match the CellFlow reproducibility repo's
    # conf/training/norman.yaml: fixed budget, no mid-training val eval)
    p.add_argument("--num-iterations", type=int, default=200_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--valid-freq", type=int, default=400_000)

    p.add_argument("--force-train", action="store_true")
    p.add_argument("--hf-repo", default=None)
    _wandb.add_args(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.dataset)
    print(f"==> dataset: {manifest['name']}")

    inputs = load_dataset(manifest, args)
    cf, train_stats = maybe_train(inputs, args.dataset, args)

    pert_col = manifest["obs"]["pert_col"]
    control_label = manifest["obs"]["control_label"]
    preds = predict(cf, inputs)

    output = args.output or (
        REPO_ROOT / "predictions" / f"cellflow_{args.dataset}_{args.split}.h5ad"
    )
    save_predictions(
        preds, inputs, output, args.dataset, args.split, train_stats, pert_col, control_label
    )


if __name__ == "__main__":
    main()
