# Data Explorer Agent -- NVIDIA Launchable

**#1 on the DABStep Leaderboard** | Built on NVIDIA NeMo Agent Toolkit (NAT)

Data Explorer is an AI-powered agent for automated data analysis. It achieved state-of-the-art performance on the [DABStep benchmark](https://huggingface.co/datasets/adyen/DABstep) (Data Agent Benchmark for Multi-step Reasoning), outperforming solutions from AntGroup and Google AI with a **30x speedup** over the Claude Code baseline.

## Quick Start

### 1. Get an API Key

You need an NVIDIA Inference API key. Get one for free at [build.nvidia.com](https://build.nvidia.com).

### 2. Run with Docker

```bash
# Build the container
docker build -t data-explorer-agent .

# Run with your API key
docker run -it --rm \
    -p 8888:8888 \
    -p 8000:8000 \
    -e NV_INFER_API_KEY=your_key_here \
    data-explorer-agent
```

Open http://localhost:8888 in your browser.

### 3. Explore the Demo Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/demo_dabstep.ipynb` | **DABStep Inference** -- The leaderboard-winning agent on financial payments questions, with before/after comparison |
| `notebooks/demo_qa.ipynb` | **General Tabular QA** -- Ask questions about any dataset with no prior knowledge |
| `notebooks/demo_eda.ipynb` | **Open-ended EDA** -- Upload any tabular data and get an automated analysis notebook with visualizations |

## What's Inside

### Open-Ended Exploratory Data Analysis

Give the agent any CSV, Parquet, or JSON file and an optional prompt. It generates a complete Jupyter notebook with data analysis, statistical summaries, and visualizations -- iteratively refining plots using vision model feedback.

- **LLM**: Nemotron 3 Super via NVIDIA Inference Hub
- **Workflow**: ReAct agent with notebook manipulation tools
- **Output**: A self-contained Jupyter notebook

### DABStep Benchmark Inference

The agent answers complex, multi-step questions about a financial payments dataset. It uses pre-distilled domain knowledge (`helper.py`) and few-shot examples to solve tasks in seconds.

- **LLM**: Nemotron 3 Super via NVIDIA Inference Hub
- **Workflow**: Tool-calling agent with stateful Python executor
- **Output**: Answer + full reasoning trace

### The Three-Phase Approach

1. **Learning** (offline): A heavyweight model solves training tasks and distills reusable functions into `helper.py`
2. **Inference** (fast): A lightweight model uses the distilled code to solve new tasks in seconds
3. **Reflection** (offline): Quality control via self-consistency checks, feeding insights back into inference

## Architecture

```
┌──────────────────────────────────────────────────┐
│                  NVIDIA Launchable                │
│                                                   │
│  JupyterLab (port 8888)                          │
│    ├── demo_eda.ipynb      → EDA Workflow         │
│    └── demo_dabstep.ipynb  → FastAPI Server       │
│                                                   │
│  DABStep Server (port 8000)                       │
│    └── POST /solve  → Tool-calling Agent          │
│                        └── helper.py (distilled)  │
│                                                   │
│  Data: payments.csv, fees.json, merchant_data...  │
└──────────────────────────────────────────────────┘
          │                    │
          ▼                    ▼
   NVIDIA Inference Hub   NVIDIA Inference Hub
   (Nemotron 3 Super)    (Nemotron 3 Super)
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `NV_INFER_API_KEY` | Yes | NVIDIA Inference Hub API key |
| `JUPYTER_TOKEN` | No | JupyterLab access token (empty = no auth) |

## Run Without Docker

```bash
# Install dependencies
uv sync --all-groups

# Download DABStep data
uv run python dabstep_agent/download_data.py

# Set API key
export NV_INFER_API_KEY=your_key_here

# Start the inference server
uv run python dabstep_agent/inference/server.py &

# Start JupyterLab
uv run jupyter lab --port 8888
```

## Team

Built by the KGMON-LLM Agent Research Team at NVIDIA:

- Jiwei Liu (jiweil@nvidia.com)
- Maximilian Jeblick (mjeblick@nvidia.com)
- Jack Yu (jacyu@nvidia.com)
