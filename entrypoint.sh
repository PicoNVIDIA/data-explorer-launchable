#!/usr/bin/env bash
set -e

if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "Starting DABStep Inference Server on port 8000..."
    uv run python dabstep_agent/inference/server.py --port 8000 &
    SERVER_PID=$!
else
    echo "WARNING: ANTHROPIC_API_KEY not set. DABStep server will NOT start."
    echo "The demo_dabstep notebook will not work without it."
    echo "Get a key at https://console.anthropic.com"
fi

echo "Starting JupyterLab on port 8888..."
exec uv run jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --NotebookApp.token="${JUPYTER_TOKEN}" \
    --ServerApp.root_dir=/app \
    --LabApp.default_url='/lab/tree/notebooks/START_HERE.ipynb'
