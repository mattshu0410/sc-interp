#!/usr/bin/env bash
# Download the State-Replogle-Filtered dataset from HuggingFace.
# Repo: arcinstitute/State-Replogle-Filtered (~30 GB)
#
# Brings down replogle_concat.h5ad (4 cell lines, State paper preprocessing)
# plus the .toml split definitions. Long-running — wrap in tmux:
#   tmux new -s replogle-dl './data/download_replogle.sh'

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_VENV="$REPO_ROOT/tools/.venv"
LOCAL_DIR="$REPO_ROOT/data/replogle"

if [ ! -d "$TOOLS_VENV" ]; then
    echo "error: tools venv not found at $TOOLS_VENV"
    echo "run ./tools/setup.sh first"
    exit 1
fi

source "$TOOLS_VENV/bin/activate"

python3 - <<EOF
from huggingface_hub import snapshot_download
local_dir = snapshot_download(
    repo_id="arcinstitute/State-Replogle-Filtered",
    repo_type="dataset",
    local_dir="$LOCAL_DIR",
)
print(f"downloaded to {local_dir}")
EOF
