#!/usr/bin/env bash
# Setup environment for CellFlow
# Called from the repo root: ./models/setup_cellflow.sh

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/cellflow" && pwd)"
cd "$MODEL_DIR"

echo "==> Setting up CellFlow environment..."

# Create venv pinned to Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install cellflow in editable mode with optional extras
# (embedding/external/pp are needed because cellflow's __init__ imports them eagerly)
uv pip install -e ".[embedding,external,pp]"

# Experiment tracking. omegaconf is needed by cellflow's WandbLogger.
uv pip install wandb omegaconf

# Replace CPU JAX with CUDA 13 build for GPU support
uv pip install --upgrade "jax[cuda13]"

echo "==> CellFlow environment ready at $MODEL_DIR/.venv"
