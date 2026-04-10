#!/usr/bin/env bash
# One-shot setup for sc-interp research repo.
# Clone the repo, then run: ./setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Install GitHub CLI if not present ────────────────────────────────────────
if ! command -v gh &>/dev/null; then
    echo "==> Installing gh..."
    sudo apt update -qq
    sudo apt install -y gh
fi

# ── Install Claude Code if not present ───────────────────────────────────────
if ! command -v claude &>/dev/null; then
    echo "==> Installing Claude Code..."
    curl -fsSL https://claude.ai/install.sh | bash
fi

# ── Install peonping if not present ──────────────────────────────────────────
if ! command -v peonping &>/dev/null; then
    echo "==> Installing peonping..."
    curl -fsSL https://peonping.com/install | bash
fi

# ── Install uv if not present ────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv $(uv --version)"

# ── Init submodules ──────────────────────────────────────────────────────────
echo "==> Initializing git submodules..."
git submodule update --init --recursive

# ── Setup each model environment ─────────────────────────────────────────────
for setup_script in models/setup_*.sh; do
    echo ""
    echo "================================================================"
    echo "  Running: $setup_script"
    echo "================================================================"
    bash "$setup_script"
done

echo ""
echo "==> All done. Each model has its own .venv under models/<name>/.venv"
