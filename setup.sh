#!/usr/bin/env bash
# Brev setup script for Data Explorer Launchable
# Usage: brev start https://github.com/PicoNVIDIA/data-explorer-launchable --setup-path setup.sh
set -e

echo "=== Data Explorer Agent Setup ==="

# Install uv if not present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || true
fi

# Find the repo directory (Brev clones into ~/verb-workspace or similar)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_DIR"
echo "Working directory: $REPO_DIR"

# Install Python dependencies
echo "Installing dependencies..."
uv sync --all-groups --frozen

# Download DABStep data from HuggingFace
echo "Downloading DABStep data..."
uv run python dabstep_agent/download_data.py

# Generate file structures cache
echo "Generating data schema cache..."
uv run python dabstep_agent/generate_file_structures.py

# Start DABStep inference server (if NVIDIA key is set)
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "Starting DABStep Inference Server on port 8000..."
    uv run python dabstep_agent/inference/server.py --port 8000 &
else
    echo "WARNING: ANTHROPIC_API_KEY not set. DABStep server will NOT start."
    echo "Get a free key at https://build.nvidia.com"
fi

# Start JupyterLab
echo "Starting JupyterLab on port 8888..."
exec uv run jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token="" \
    --ServerApp.root_dir="$REPO_DIR" \
    --LabApp.default_url='/lab/tree/notebooks/START_HERE.ipynb'
