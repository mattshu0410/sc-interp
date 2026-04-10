#!/usr/bin/env bash
# Setup environment for scLDM
# Called from the repo root: ./models/setup_scldm.sh

set -euo pipefail

MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scldm" && pwd)"
cd "$MODEL_DIR"

echo "==> Setting up scLDM environment..."

# Create venv pinned to Python 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate

# Install scldm in editable mode + cellarium-ml from source
# (PyPI cellarium-ml 0.0.7 is incompatible with anndata>=0.10.9)
# Pin scvi-tools<1.3 to avoid anndata.io import breakage with anndata 0.10.9
uv pip install -e . \
    "cellarium-ml @ git+https://github.com/cellarium-ai/cellarium-ml.git" \
    "scvi-tools>=1.2,<1.3"

# Download model artifacts from S3
echo "==> Downloading scLDM artifacts..."
scldm-download-artifacts --group resubmission

echo "==> scLDM environment ready at $MODEL_DIR/.venv"
