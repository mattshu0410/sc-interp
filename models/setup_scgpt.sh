#!/usr/bin/env bash
# Setup environment for scGPT (inference-only for perturbation prediction)
# Called from the repo root: ./models/setup_scgpt.sh
#
# Upstream scgpt depends on torchtext.vocab, which has been archived. We carry
# a torchtext-free GeneVocab replacement at models/scgpt_patches/gene_tokenizer.py
# and overlay it into the submodule below — this lets us track upstream cleanly
# and frees torch from the 2.3.0 pin, so nnsight installs without --no-deps.
# scgpt itself is still installed --no-deps to bypass the scvi-tools<1.0 pin.

set -euo pipefail

PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scgpt_patches" && pwd)"
MODEL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/scgpt" && pwd)"
cd "$MODEL_DIR"

echo "==> Setting up scGPT environment..."

# Overlay the gene_tokenizer.py shim onto the submodule. Idempotent — the cp
# just rewrites the same bytes if already applied. Re-run this script after
# any `git submodule update` to restore the patch (submodule update resets
# the working tree).
cp "$PATCH_DIR/gene_tokenizer.py" "$MODEL_DIR/scgpt/tokenizer/gene_tokenizer.py"
echo "==> applied patch: scgpt/tokenizer/gene_tokenizer.py"

uv venv --python 3.11 --clear .venv
source .venv/bin/activate

uv pip install torch torchvision

uv pip install \
    "numpy<2" \
    "pandas" \
    "anndata" \
    "scanpy" \
    "scikit-misc" \
    "numba" \
    "umap-learn" \
    "leidenalg" \
    "ipython" \
    "datasets" \
    "wandb"

# Pinned to 0.1.2 to match tools/.venv: scgpt and the materialiser share
# data/<dataset>/cell_graphs.pkl, and the on-disk Data layout differs across
# the 0.0.x → 0.1.x boundary (1-col x + Data.pert_idx vs 2-col x). The
# runner's _load_gears patches x to the 2-col layout in memory.
uv pip install "cell-gears==0.1.2" "torch-geometric"

# --no-deps to bypass scvi-tools<1.0, orbax<0.1.8, etc.
uv pip install --no-deps -e .

# Activation capture
uv pip install "nnsight>=0.5,<0.6" "h5py"

uv pip install gdown

# ── Download the whole-human checkpoint ──────────────────────────────────────
# Skip by setting SCGPT_SKIP_CHECKPOINT=1.
CHECKPOINT_DIR="$MODEL_DIR/checkpoints/scGPT_human"
GDRIVE_FOLDER="https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y"

if [ "${SCGPT_SKIP_CHECKPOINT:-}" = "1" ]; then
    echo "==> SCGPT_SKIP_CHECKPOINT=1, skipping checkpoint download"
elif [ -f "$CHECKPOINT_DIR/best_model.pt" ] && \
     [ -f "$CHECKPOINT_DIR/args.json" ] && \
     [ -f "$CHECKPOINT_DIR/vocab.json" ]; then
    echo "==> scGPT whole-human checkpoint already present at $CHECKPOINT_DIR"
else
    echo "==> Downloading scGPT whole-human checkpoint to $CHECKPOINT_DIR..."
    mkdir -p "$CHECKPOINT_DIR"
    gdown --folder "$GDRIVE_FOLDER" -O "$CHECKPOINT_DIR"
fi

echo "==> scGPT environment ready at $MODEL_DIR/.venv"
echo "    checkpoint:  $CHECKPOINT_DIR"
