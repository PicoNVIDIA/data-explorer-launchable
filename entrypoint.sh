#!/usr/bin/env bash
set -e

echo "Starting DABStep Inference Server on port 8000..."
uv run python dabstep_agent/inference/server.py --port 8000 &
SERVER_PID=$!

echo "Starting JupyterLab on port 8888..."
exec uv run jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token="${JUPYTER_TOKEN}" \
    --ServerApp.root_dir=/app
