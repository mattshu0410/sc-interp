#!/usr/bin/env bash
# Download the Norman 2019 Perturb-seq dataset and materialise its canonical
# train/val/test split. Runs in the tools venv (gears lives there). GEARS
# pulls the raw dataset from Harvard Dataverse on first run and caches it
# under data/norman/; the materialiser then computes the simulation split
# and writes it as canonical JSON for runners to consume.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Use the gears model venv, not tools/.venv: tools installs `cell-gears` from
# PyPI (0.1.2, Dec 2023), whose create_cell_graph omits `pert_idx` on each
# Data. model.forward reads `data.pert_idx`, so the pkl it writes is unusable.
# The submodule is the source of truth — share it.
GEARS_VENV="$REPO_ROOT/models/gears/.venv"

if [ ! -d "$GEARS_VENV" ]; then
    echo "error: gears venv not found at $GEARS_VENV"
    echo "run ./models/setup_gears.sh first"
    exit 1
fi

source "$GEARS_VENV/bin/activate"
cd "$REPO_ROOT"

python -m scripts.data.gears --dataset norman
