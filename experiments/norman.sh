#!/usr/bin/env bash
# Reproduce the Norman 2019 benchmark by pulling our fine-tuned weights
# from HuggingFace and running inference + cell-eval. Assumes ./setup.sh
# has been run. Takes ~25 minutes on a fresh VM.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

bash data/download_norman.sh

source models/scgpt/.venv/bin/activate
python -m scripts.run scgpt --dataset norman --hf-repo matthewshu/scGPT-norman-ft
deactivate

source models/cellflow/.venv/bin/activate
python -m scripts.run cellflow --dataset norman --hf-repo matthewshu/cellflow-norman-esm3b
deactivate

source tools/.venv/bin/activate
python -m scripts.analyse.eval_cell_eval --predictions predictions/scgpt_norman_test.h5ad --profile full
python -m scripts.analyse.eval_cell_eval --predictions predictions/cellflow_norman_test.h5ad --profile full
deactivate
