#!/usr/bin/env bash
# Regenerate paired scGPT activations on Norman test split: base + finetuned.
# Both runs share the same gears split (default seed=42), so the resulting
# shards align cell-for-cell on the new string cell_id (= adata obs.index).
#
# Estimated runtime: ~1-2h per side on a single GPU. Sequential, not parallel
# -- both checkpoints don't fit in memory at once for typical hosts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$REPO_ROOT/models/scgpt/.venv/bin/python"
PRETRAINED="$REPO_ROOT/models/scgpt/checkpoints/scGPT_human"
BASE_OUT="$REPO_ROOT/data/scgpt-base-norman-activations"
FT_OUT="$REPO_ROOT/data/scgpt-norman-activations"

mkdir -p "$REPO_ROOT/logs"

# H5ActivationSink mode='x' refuses overwrite; wipe the dirs to allow rerun.
rm -rf "$BASE_OUT" "$FT_OUT"

cd "$REPO_ROOT"

echo "==> $(date) :: base (--skip-finetune) -> $BASE_OUT"
"$PYTHON" -m scripts.run scgpt \
    --dataset norman --split test \
    --skip-finetune \
    --pretrained-dir "$PRETRAINED" \
    --eval-batch-size 64 \
    --capture-activations \
    --batches-per-shard 4 \
    --activation-out "$BASE_OUT" \
    --wandb-mode disabled

echo "==> $(date) :: finetuned -> $FT_OUT"
"$PYTHON" -m scripts.run scgpt \
    --dataset norman --split test \
    --pretrained-dir "$PRETRAINED" \
    --eval-batch-size 64 \
    --capture-activations \
    --batches-per-shard 4 \
    --activation-out "$FT_OUT" \
    --wandb-mode disabled

echo "==> $(date) :: done"
