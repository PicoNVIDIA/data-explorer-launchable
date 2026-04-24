# DABStep Agent

Agent pipeline for the [DABStep](https://huggingface.co/datasets/adyen/DABstep) tabular data QA benchmark. The workflow has three stages: **download data**, **learn** (discover solutions + distill into reusable code), and **inference** (solve tasks at scale).

## Directory Structure

```
dabstep_agent/
├── download_data.py           # Download DABStep dataset from HuggingFace
├── learn/                     # Learning & distillation pipeline
│   ├── learn.py               # Phase 1 – Learn solutions from scratch
│   ├── learning_config.yml
│   ├── distill_nat/           # Phase 2 – Distill via NAT runtime
│   └── distill_agent_sdk/     # Phase 2 – Distill via Claude Agent SDK
├── inference/                 # Production inference pipeline
│   ├── server.py              # FastAPI /solve endpoint
│   ├── client.py              # HTTP client for the server
│   ├── solve.py               # Core solver (prompt, extraction, tracing)
│   ├── postprocess.py         # Clean up JSONL result files
│   ├── dabstep_config.yml     # NAT workflow config for inference
│   └── new_solutions.md       # Few-shot examples for the solver prompt (populated by distill)
└── workspace/                 # Runtime artifacts (traces, generated code)
```

## Quick Start

All commands below should be run from the root of the repository (`data-explorer-agent/`).

### 1. Download Data

```bash
uv run python dabstep_agent/download_data.py
```

Downloads context files (`payments.csv`, `fees.json`, `manual.md`, etc.) and task splits (`tasks.json`, `tasks_dev.json`) from HuggingFace into `data/`.

### 2. Learn

Discover solutions from scratch against ground truth, then distill the traces into a reusable `helper.py` and `solutions.md`.

```bash
# Run learning on selected tasks
uv run python dabstep_agent/learn/learn.py --input data/tasks_dev.json --task-id 49,50,51

# Distill traces into helper code (NAT)
uv run python dabstep_agent/learn/distill_nat/distill.py

# Or distill with Claude Agent SDK
python dabstep_agent/learn/distill_agent_sdk/run_distill.py
```

See [`learn/README.md`](learn/README.md) for full details.

### 3. Inference

Solve tasks using the distilled helper code and few-shot examples.

```bash
# Start the API server
uv run python dabstep_agent/inference/server.py

# Send tasks via client
uv run python dabstep_agent/inference/client.py --input data/tasks.json

# Or run the solver directly (no server)
uv run python dabstep_agent/inference/solve.py --input data/tasks.json --task-id 1712,1810

# Post-process results
uv run python dabstep_agent/inference/postprocess.py results.jsonl results_clean.jsonl
```

See [`inference/README.md`](inference/README.md) for full details.

## Environment

Requires `$ANTHROPIC_API_KEY` set in your environment or `.env` file. Get a key at https://console.anthropic.com.
