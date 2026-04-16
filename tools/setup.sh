#!/usr/bin/env bash
# Setup the shared tools venv used for data acquisition, evaluation,
# and other non-model utilities (gears, cell-eval, etc).
# Not tied to any specific model.

set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$TOOLS_DIR"

echo "==> Setting up tools venv..."

uv venv --python 3.12 --clear .venv
source .venv/bin/activate

# CPU torch is enough for downloads and data processing.
uv pip install --index-url https://download.pytorch.org/whl/cpu "torch"

# GEARS (PertData loader for Norman, Adamson, Replogle, Dixit datasets)
# plus its transitive needs that aren't auto-installed
uv pip install \
    "cell-gears<0.0.3" \
    "torch-geometric" \
    "anndata" \
    "scanpy"

# cell-eval is the authoritative benchmark metric set used by
# scripts/analyse/eval_cell_eval.py and scripts/analyse/paper_metrics.py
uv pip install cell-eval

echo "==> tools venv ready at $TOOLS_DIR/.venv"
