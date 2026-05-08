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
# faiss-gpu-cu12 backs torchdr's neighbour search (UMAP / LargeVis / PaCMAP).
# The legacy `faiss-gpu` PyPI package is deprecated; this is the CUDA-12 build.
uv pip install faiss-gpu-cu12
uv pip install hydra-core omegaconf
uv pip install scikit-learn h5py numpy einops
uv pip install pytest wandb
# dictionary_learning ships CrossCoder/BatchTopKCrossCoder/BatchTopKSAE +
# their trainers + trainSAE. We use submodule imports only; the package
# __init__ pulls in nnsight which we never call into.
uv pip install "git+https://github.com/science-of-finetuning/crosscoder_learning.git"

echo "==> diffing venv ready at $DIFF_DIR/.venv"
