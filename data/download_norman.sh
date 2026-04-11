#!/usr/bin/env bash
# Download the Norman 2019 Perturb-seq dataset and materialize its canonical
# train/val/test split. Runs in the tools venv (gears lives there). GEARS
# pulls the raw dataset from Harvard Dataverse on first run and caches it
# under data/norman/; the materializer then computes the simulation split
# and writes it as canonical JSON for runners to consume.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS_VENV="$REPO_ROOT/tools/.venv"

if [ ! -d "$TOOLS_VENV" ]; then
    echo "error: tools venv not found at $TOOLS_VENV"
    echo "run ./tools/setup.sh first"
    exit 1
fi

source "$TOOLS_VENV/bin/activate"
cd "$REPO_ROOT"

python -m scripts.data.gears --dataset norman
