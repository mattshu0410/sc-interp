#!/usr/bin/env bash
# Full Norman extraction: all splits (train, val, test) @ float32.
#
# Run after 20260414_extract_scgpt_calibration.sh has finished uploading.
# Having all splits enables MI comparisons across conditions that span
# different splits (val vs test perturbation conditions etc).
#
# Output directories:
#   data/activations/norman_scgpt_full-train/
#   data/activations/norman_scgpt_full-val/
#   data/activations/norman_scgpt_full-test/
#
# HF dataset repos:
#   kevinychou/scgpt-activations-norman-full-train
#   kevinychou/scgpt-activations-norman-full-val
#   kevinychou/scgpt-activations-norman-full-test
#
# Rough estimates @ float32:
#   extraction: ~35 min/split (GPU-bound, ~43 cells/s)
#   upload:      ~6 hr/split  (network-bound, ~45 MB/s)
#   total:      ~20 hr for all three splits

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash data/download_norman.sh

source models/scgpt/.venv/bin/activate

for split in train val test; do
    echo ""
    echo "========================================"
    echo "==> split: ${split}"
    echo "========================================"
    python -m scripts.run extract-scgpt \
        --dataset norman \
        --split "${split}" \
        --dtype float32 \
        --n-bins 51 \
        --tag "full-${split}" \
        --hf-repo "kevinychou/scgpt-activations-norman-full-${split}"

    python -m scripts.tests.validate_scgpt_activations \
        --dataset norman \
        --split "${split}" \
        --n-bins 51 \
        --tag "full-${split}"
done

deactivate
echo ""
echo "==> full extraction complete for all splits"
