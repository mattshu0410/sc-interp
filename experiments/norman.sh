#!/usr/bin/env bash
# Reproduce the Norman 2019 Perturb-seq perturbation-prediction benchmark.
#
# Assumes ./setup.sh has already been run so tools, docs, and model venvs
# exist. Each step checks for its own output and skips if already done, so
# the script is idempotent and safe to rerun. Does NOT do HF uploads or
# wandb login; those are manual.
#
# Usage:
#     ./experiments/norman.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

section() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
}

# ── Step 1: download and preprocess Norman via GEARS ─────────────────────────
# Writes data/norman/perturb_processed.h5ad and the split pickle under
# data/norman/splits/. Uses the tools venv (gears lives there).
if [ -f "data/norman/perturb_processed.h5ad" ]; then
    section "step 1: Norman data already present, skipping"
else
    section "step 1: downloading Norman via GEARS"
    bash data/download_norman.sh
fi

# ── Step 2: train scGPT and write prediction h5ad ────────────────────────────
# Caches the fine-tuned weights under models/scgpt/checkpoints/norman_ft/
# and writes predictions/scgpt_norman_test.h5ad. Skips training if the
# cached checkpoint already exists.
if [ -f "predictions/scgpt_norman_test.h5ad" ]; then
    section "step 2: scGPT predictions already present, skipping"
else
    section "step 2: fine-tuning scGPT and predicting"
    source models/scgpt/.venv/bin/activate
    python -m scripts.run scgpt \
        --dataset norman \
        --split test \
        --num-epochs 15 \
        --early-stop 10 \
        --stop-metric pearson
    deactivate
fi

# ── Step 3: score scGPT predictions with cell-eval ───────────────────────────
# Writes eval_outputs/scgpt_norman_test/{results,agg_results}.csv using the
# full profile (pearson_delta, discrimination_score_*, pearson_edistance,
# overlap_at_N, DE metrics, clustering_agreement).
if [ -d "eval_outputs/scgpt_norman_test" ]; then
    section "step 3: scGPT eval_outputs already present, skipping"
else
    section "step 3: running cell-eval on scGPT predictions"
    source tools/.venv/bin/activate
    python -m scripts.analyze.eval_cell_eval \
        --predictions predictions/scgpt_norman_test.h5ad \
        --profile full
    deactivate
fi

# ── Step 4: train CellFlow and write prediction h5ad ─────────────────────────
# Caches weights under models/cellflow/checkpoints/norman/CellFlow.pkl and
# writes predictions/cellflow_norman_test.h5ad. Matches the paper's Norman
# config (50-dim PCA, architecture from cellflow_reproducibility repo,
# 200k iterations).
if [ -f "predictions/cellflow_norman_test.h5ad" ]; then
    section "step 4: CellFlow predictions already present, skipping"
else
    section "step 4: training CellFlow and predicting"
    source models/cellflow/.venv/bin/activate
    python -m scripts.run cellflow \
        --dataset norman \
        --num-iterations 200000 \
        --batch-size 1024 \
        --valid-freq 400000
    deactivate
fi

# ── Step 5: score CellFlow predictions with cell-eval ────────────────────────
if [ -d "eval_outputs/cellflow_norman_test" ]; then
    section "step 5: CellFlow eval_outputs already present, skipping"
else
    section "step 5: running cell-eval on CellFlow predictions"
    source tools/.venv/bin/activate
    python -m scripts.analyze.eval_cell_eval \
        --predictions predictions/cellflow_norman_test.h5ad \
        --profile full
    deactivate
fi

# ── Summary ──────────────────────────────────────────────────────────────────
section "done"
echo ""
echo "Prediction h5ads:"
echo "  predictions/scgpt_norman_test.h5ad"
echo "  predictions/cellflow_norman_test.h5ad"
echo ""
echo "Aggregated cell-eval metrics:"
echo "  eval_outputs/scgpt_norman_test/agg_results.csv"
echo "  eval_outputs/cellflow_norman_test/agg_results.csv"
