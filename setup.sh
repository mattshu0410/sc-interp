#!/usr/bin/env bash
# One-shot setup for sc-interp research repo.
# Clone the repo, then run: ./setup.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ── Check / install NVIDIA driver ────────────────────────────────────────────
# We need driver >= 580 for CUDA 13 (jax[cuda13], current PyTorch, etc).
# This step requires a reboot, so if we install we exit and ask the user
# to reboot then re-run.
MIN_DRIVER_MAJOR=580
if command -v nvidia-smi &>/dev/null; then
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | cut -d. -f1)
    if [ -n "$DRIVER_MAJOR" ] && [ "$DRIVER_MAJOR" -lt "$MIN_DRIVER_MAJOR" ]; then
        echo "==> NVIDIA driver $DRIVER_MAJOR is older than $MIN_DRIVER_MAJOR, upgrading..."
        sudo apt update -qq
        sudo apt install -y "nvidia-driver-${MIN_DRIVER_MAJOR}" -o Dpkg::Options::=--force-overwrite
        sudo apt autoremove -y --purge
        echo ""
        echo "================================================================"
        echo "  Driver $MIN_DRIVER_MAJOR installed. Please run:"
        echo ""
        echo "      sudo reboot"
        echo ""
        echo "  Then re-run ./setup.sh to finish the rest of the install."
        echo "================================================================"
        exit 0
    fi
fi

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

# ── Install uv if not present ────────────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in this session
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv $(uv --version)"

# ── Init submodules ──────────────────────────────────────────────────────────
echo "==> Initialising git submodules..."
git submodule update --init --recursive

# ── Setup the shared tools venv (gears, cell-eval, ...) ──────────────────────
echo ""
echo "================================================================"
echo "  Running: tools/setup.sh"
echo "================================================================"
bash tools/setup.sh

# ── Setup the docs venv (docling for paper conversion) ───────────────────────
echo ""
echo "================================================================"
echo "  Running: docs/setup.sh"
echo "================================================================"
bash docs/setup.sh

# ── Setup each model environment ─────────────────────────────────────────────
for setup_script in models/setup_*.sh; do
    echo ""
    echo "================================================================"
    echo "  Running: $setup_script"
    echo "================================================================"
    bash "$setup_script"
done

echo ""
echo "================================================================"
echo "  All done."
echo "================================================================"
echo "    tools venv:  tools/.venv"
echo "    docs venv:   docs/.venv"
echo "    model venvs: models/<name>/.venv"
echo ""
echo "  Two manual steps remain before you can run experiments:"
echo ""
echo "    1. wandb login     (needs an API key from https://wandb.ai/authorize)"
echo "    2. gh auth login   (needs a GitHub token, for git push over HTTPS)"
echo ""
echo "  Both persist to ~/.netrc / ~/.config/gh/, and only need to"
echo "  be done once per VM. See the sc-interp README for more detail."
