#!/usr/bin/env bash
# Setup docs venv with docling for converting papers to markdown.
set -euo pipefail

DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DOCS_DIR"

echo "==> Setting up docs venv..."
uv venv --python 3.12 --clear .venv
source .venv/bin/activate
uv pip install docling paperscraper
echo "==> docs venv ready at $DOCS_DIR/.venv"
