#!/usr/bin/env bash
# Convert a paper (local PDF or URL) to markdown via docling.
# Usage:
#   ./docs/add_paper.sh <pdf_url> <output_name>
#   ./docs/add_paper.sh <local_pdf_path> <output_name>
set -euo pipefail

if [ $# -ne 2 ]; then
    echo "usage: $0 <pdf_path_or_url> <output_name>"
    exit 1
fi

SRC="$1"
NAME="$2"
DOCS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAPERS_DIR="$DOCS_DIR/papers"
mkdir -p "$PAPERS_DIR"

PDF="$PAPERS_DIR/$NAME.pdf"

if [[ "$SRC" =~ ^https?:// ]]; then
    echo "==> downloading $SRC"
    curl -fSL -A "Mozilla/5.0" -o "$PDF" "$SRC"
else
    echo "==> copying $SRC"
    cp "$SRC" "$PDF"
fi

source "$DOCS_DIR/.venv/bin/activate"
echo "==> converting to markdown"
docling --to md --output "$PAPERS_DIR" "$PDF"

echo "==> done"
ls -la "$PAPERS_DIR/$NAME".*
