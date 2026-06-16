#!/usr/bin/env bash
# Build a scGPT-vocab-aligned ESM2 gene prior from arcinstitute/SE-600M.
#
# Pulls protein_embeddings.pt (19,790 genes × 5120 dims, ESM2-15B mean-pooled
# per protein-coding gene) and remaps it to scGPT's vocabulary, producing a
# [vocab_size, 5120] tensor saved as safetensors. Genes in the vocab without
# an ESM entry (special tokens, non-coding RNAs, pseudogenes) get zero rows;
# GenePriorEncoder treats those as "no prior" via its zero-row lookup.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_VENV="$REPO_ROOT/tools/.venv"
VOCAB_PATH="$REPO_ROOT/models/scgpt/checkpoints/scGPT_human/vocab.json"
OUT_PATH="$REPO_ROOT/data/scgpt_esm_prior.safetensors"
HF_CACHE="$REPO_ROOT/cache/huggingface/hub"

if [ ! -d "$TOOLS_VENV" ]; then
    echo "error: tools venv not found at $TOOLS_VENV"
    echo "run ./tools/setup.sh first"
    exit 1
fi
if [ ! -f "$VOCAB_PATH" ]; then
    echo "error: scGPT vocab not found at $VOCAB_PATH"
    echo "run ./models/setup_scgpt.sh first"
    exit 1
fi

source "$TOOLS_VENV/bin/activate"

VOCAB_PATH="$VOCAB_PATH" OUT_PATH="$OUT_PATH" HF_CACHE="$HF_CACHE" python3 - <<'EOF'
import json
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file

vocab_path = os.environ["VOCAB_PATH"]
out_path = os.environ["OUT_PATH"]
hf_cache = os.environ["HF_CACHE"]

print("Downloading arcinstitute/SE-600M/protein_embeddings.pt ...")
src = hf_hub_download(
    repo_id="arcinstitute/SE-600M",
    filename="protein_embeddings.pt",
    cache_dir=hf_cache,
)

vocab = json.load(open(vocab_path))
esm = torch.load(src, map_location="cpu", weights_only=False)
prior_dim = next(iter(esm.values())).shape[0]
table_size = max(vocab.values()) + 1

print(
    f"  vocab tokens     : {len(vocab):,}  (max id {table_size - 1:,})\n"
    f"  arc esm entries  : {len(esm):,}\n"
    f"  prior dim        : {prior_dim}"
)

table = torch.zeros((table_size, prior_dim), dtype=torch.float32)
matched = 0
for symbol, idx in vocab.items():
    vec = esm.get(symbol)
    if vec is not None:
        table[idx] = vec
        matched += 1

print(
    f"  matched          : {matched:,}  ({100 * matched / len(vocab):.1f}% of vocab, "
    f"{100 * matched / len(esm):.1f}% of arc table)"
)

save_file({"embeddings": table.contiguous()}, out_path)
print(f"Wrote {out_path}  ({os.path.getsize(out_path) / 1e6:.1f} MB)")
EOF
