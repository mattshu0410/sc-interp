#!/usr/bin/env bash
# Setup environment for GEARS
# Called from the repo root: ./models/setup_gears.sh

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/gears" && pwd)"
cd "$MODEL_DIR"

echo "==> Setting up GEARS environment..."

# Create venv pinned to Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install torch first since GEARS leaves it unpinned in requirements.txt
uv pip install torch torchvision

# PyTorch Geometric. GEARS only uses torch_geometric.data and SGConv,
# so no C-extensions (torch-scatter / torch-sparse / torch-cluster) are needed.
# Pinned <2.4 because PyG 2.4 tightened Batch.from_data_list to drop Python-list
# attributes it can't collate; GEARS stores pert_idx/de_idx as plain lists on
# Data objects and model.forward reads data.pert_idx off the batch. Upstream
# GEARS is dormant, so the pin stays on our side.
uv pip install "torch-geometric<2.4"

# Install GEARS from the submodule source. The PyPI release (0.1.2) is
# from Dec 2023 and lags the GitHub main branch, so we install editable.
uv pip install -e .

# Activation capture (scripts/interp/) — nnsight wraps the model for tracing,
# h5py is the on-disk sink. nnsight 0.5.x source-AST tracer is what HookManager
# is written against; any upgrade wants a tracer-behavior re-check.
uv pip install "nnsight>=0.5,<0.6" "h5py"

# Smoke test
python -c "import gears; from gears import PertData, GEARS; print('gears import OK')"

echo "==> GEARS environment ready at $MODEL_DIR/.venv"
