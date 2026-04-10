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
uv pip install torch-geometric

# Install GEARS from the submodule source. The PyPI release (0.1.2) is
# from Dec 2023 and lags the GitHub main branch, so we install editable.
uv pip install -e .

# Smoke test
python -c "import gears; from gears import PertData, GEARS; print('gears import OK')"

echo "==> GEARS environment ready at $MODEL_DIR/.venv"
