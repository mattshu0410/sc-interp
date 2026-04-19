#!/usr/bin/env bash
# Setup environment for scGPT (inference-only for perturbation prediction)
# Called from the repo root: ./models/setup_scgpt.sh
#
# scGPT has a brittle dependency stack, so this script is deliberate:
# - Pins Python 3.11 (older than 3.12 to dodge wheel gaps in old deps)
# - Pins torch 2.3.0 + torchtext 0.18.0 (last version of torchtext before it was abandoned)
# - Skips scvi-tools/scib/datasets/tensorflow/wandb/flash-attn (not needed for inference)
# - Installs scgpt from source with --no-deps to bypass the scvi-tools<1.0 pin

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scgpt" && pwd)"
cd "$MODEL_DIR"

echo "==> Setting up scGPT environment..."

# Create venv pinned to Python 3.11
uv venv --python 3.11 --clear .venv
source .venv/bin/activate

# Torch 2.3.0 with CUDA 12.1 wheels (forward-compatible with driver 580 / CUDA 13)
uv pip install \
    "torch==2.3.0" \
    "torchvision==0.18.0" \
    --index-url https://download.pytorch.org/whl/cu121

# torchtext 0.18.0 is the last release and only exists on PyPI (not the pytorch index)
# It requires exactly torch 2.3.0 which we just pinned above
uv pip install "torchtext==0.18.0"

# Core scientific deps that scGPT's inference path actually imports
# IPython is imported at module load in scgpt/utils/util.py
uv pip install \
    "numpy<2" \
    "pandas" \
    "anndata" \
    "scanpy" \
    "scikit-misc" \
    "numba" \
    "umap-learn" \
    "leidenalg" \
    "ipython" \
    "datasets" \
    "wandb"

# GEARS provides PertData, the Norman/Adamson/Replogle loader used in the tutorial
# scGPT pins cell-gears<0.0.3, so stick with that. torch-geometric is a gears dep
# that isn't auto-installed. Core torch-geometric is pure Python, no compiled kernels needed.
uv pip install "cell-gears<0.0.3" "torch-geometric"

# Install scgpt itself from source, no-deps to skip scvi-tools<1.0, orbax<0.1.8, etc.
uv pip install --no-deps -e .

# Activation capture (scripts/interp/) — nnsight wraps the model for tracing,
# h5py is the on-disk sink. nnsight 0.5 declares torch>=2.4 but HookManager's
# trace path only uses forward hooks + autograd (stable since torch 2.0), and
# torch must stay at 2.3.0 because torchtext 0.18.0 is ABI-pinned to it. So
# --no-deps preserves the pin; runtime deps go in explicitly below (they all
# accept torch 2.3.0 or don't depend on torch).
uv pip install --no-deps "nnsight>=0.5,<0.6"
uv pip install \
    "transformers" \
    "astor" \
    "cloudpickle" \
    "python-socketio[client]" \
    "pydantic>=2.9.0" \
    "accelerate" \
    "toml" \
    "h5py"

# gdown handles the Google Drive folder download for the checkpoint
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
