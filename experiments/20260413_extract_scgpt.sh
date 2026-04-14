#!/usr/bin/env bash
# Extract scGPT activations for the Norman 2019 dataset and upload to HuggingFace.
# Assumes ./setup.sh has been run. Takes ~10 minutes on first run (data download
# + extraction); subsequent runs skip the download if data is already cached.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash data/download_norman.sh

source models/scgpt/.venv/bin/activate
python -m scripts.run extract-scgpt \
    --dataset norman \
    --split test \
    --max-cells 50 \
    --dtype float16 \
    --hf-repo kevinychou/scgpt-activations-norman
deactivate
