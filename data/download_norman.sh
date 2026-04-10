#!/usr/bin/env bash
# Download the Norman 2019 Perturb-seq dataset via GEARS PertData.
# GEARS pulls from Harvard Dataverse and lays the files out in a fixed
# structure under data/norman/ that the run scripts consume.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
TOOLS_VENV="$REPO_ROOT/tools/.venv"

if [ ! -d "$TOOLS_VENV" ]; then
    echo "error: tools venv not found at $TOOLS_VENV"
    echo "run ./tools/setup.sh first"
    exit 1
fi

source "$TOOLS_VENV/bin/activate"

DATA_DIR="$DATA_DIR" python - <<'PY'
import os
from pathlib import Path
from gears import PertData

data_dir = Path(os.environ["DATA_DIR"]).resolve()
data_dir.mkdir(parents=True, exist_ok=True)

print(f"==> Downloading Norman into {data_dir}")
pert_data = PertData(str(data_dir))
pert_data.load(data_name="norman")
print("==> Done")
print(f"    adata:         {pert_data.adata.shape}")
print(f"    perturbations: {len(pert_data.adata.obs['condition'].unique())}")
PY
