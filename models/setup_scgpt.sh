#!/usr/bin/env bash
# Setup environment for scGPT (inference-only for perturbation prediction)
# Called from the repo root: ./models/setup_scgpt.sh
#
# scgpt is installed --no-deps to bypass scvi-tools<1.0 / orbax<0.1.8 pins.
# Torchtext-free GeneVocab is upstream as of cebd6fa (vocab_compat.py).

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scgpt" && pwd)"
cd "$MODEL_DIR"

echo "==> Setting up scGPT environment..."

# Pin to 3.11.9: cpython-3.11.15 (Clang 22.1.3) has a for-loop variable
# scoping bug that breaks inspect.cleandoc and scipy._docscrape at import time.
uv venv --python 3.11.9 --clear .venv
source .venv/bin/activate

uv pip install torch torchvision

uv pip install \
    "numpy<2" \
    "pandas" \
    "anndata>=0.8,<0.10" \
    "scanpy>=1.9.1,<1.10" \
    "scikit-misc" \
    "numba>=0.56,<0.60" \
    "umap-learn" \
    "leidenalg" \
    "ipython" \
    "datasets" \
    "wandb" \
    "scipy<1.17" \
    "matplotlib>=3.7,<3.9"

# Pinned to 0.1.2 to match tools/.venv: scgpt and the materialiser share
# data/<dataset>/cell_graphs.pkl, and the on-disk Data layout differs across
# the 0.0.x → 0.1.x boundary (1-col x + Data.pert_idx vs 2-col x). The
# runner's _load_gears patches x to the 2-col layout in memory.
uv pip install "cell-gears==0.1.2" "torch-geometric"

# --no-deps to bypass scvi-tools<1.0, orbax<0.1.8, etc.
uv pip install --no-deps -e .

# Activation capture
uv pip install "nnsight>=0.5,<0.6" "h5py"

uv pip install gdown

# ── Download the whole-human checkpoint ──────────────────────────────────────
# Skip by setting SCGPT_SKIP_CHECKPOINT=1.
CHECKPOINT_DIR="$MODEL_DIR/checkpoints/scGPT_human"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"

if [ "${SCGPT_SKIP_CHECKPOINT:-}" = "1" ]; then
    echo "==> SCGPT_SKIP_CHECKPOINT=1, skipping checkpoint download"
elif [ -f "$CHECKPOINT_DIR/best_model.pt" ] && \
     [ -f "$CHECKPOINT_DIR/args.json" ] && \
     [ -f "$CHECKPOINT_DIR/vocab.json" ]; then
    echo "==> scGPT whole-human checkpoint already present at $CHECKPOINT_DIR"
else
    echo "==> Downloading scGPT whole-human checkpoint to $CHECKPOINT_DIR..."
    mkdir -p "$CHECKPOINT_DIR"
    gdown --folder "$GDRIVE_FOLDER" -O "$CHECKPOINT_DIR"
fi

echo "==> scGPT environment ready at $MODEL_DIR/.venv"
echo "    checkpoint:  $CHECKPOINT_DIR"
