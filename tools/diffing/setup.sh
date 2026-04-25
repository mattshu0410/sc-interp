#!/usr/bin/env bash
# Diffing venv: GPU torch, torchdr, hydra.

set -euo pipefail

DIFF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIFF_DIR"

echo "==> Setting up diffing venv..."

uv venv --python 3.12 --clear .venv
source .venv/bin/activate

uv pip install torch torchvision
uv pip install torchdr matplotlib
uv pip install hydra-core omegaconf
uv pip install scikit-learn h5py numpy
uv pip install pytest wandb

echo "==> diffing venv ready at $DIFF_DIR/.venv"
