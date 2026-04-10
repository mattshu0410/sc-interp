#!/usr/bin/env bash
# Download datasets by URL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASETS=(
    "https://datasets.cellxgene.cziscience.com/9deda9ad-6a71-401e-b909-5263919d85f9.h5ad"
)

for url in "${DATASETS[@]}"; do
    dest="$(basename "$url")"
    if [ -f "$dest" ]; then
        echo "==> $dest exists, skipping"
        continue
    fi
    echo "==> Downloading $dest..."
    curl -fSL -o "$dest" "$url"
done

echo "==> Done"
