#!/usr/bin/env bash
# Calibration run: 1000 cells @ float32 from the Norman test split.
#
# Purpose: measure real extraction + upload throughput on this machine
# before committing to the full test split (~10k cells @ float32 ≈ 390 GB).
#
# Isolation: uses --tag calib-1k-f32 so output lives at
#   data/activations/norman_scgpt_calib-1k-f32/
# and uploads to a distinct HF dataset repo, so it won't collide with
# the smoke-test extraction at data/activations/norman_scgpt/.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash data/download_norman.sh

source models/scgpt/.venv/bin/activate
python -m scripts.run extract-scgpt \
    --dataset norman \
    --split test \
    --max-cells 1000 \
    --dtype float32 \
    --n-bins 51 \
    --tag calib-1k-f32 \
    --hf-repo kevinychou/scgpt-activations-norman-calib-1k-f32

python -m scripts.tests.validate_scgpt_activations \
    --dataset norman \
    --split test \
    --n-bins 51 \
    --tag calib-1k-f32
deactivate
