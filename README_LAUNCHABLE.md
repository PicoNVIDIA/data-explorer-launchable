# Data Explorer Agent -- NVIDIA Launchable

**#1 on the DABStep Leaderboard** | Built on NVIDIA NeMo Agent Toolkit (NAT)

Data Explorer is an AI-powered agent for automated data analysis. It achieved state-of-the-art performance on the [DABStep benchmark](https://huggingface.co/datasets/adyen/DABstep) (Data Agent Benchmark for Multi-step Reasoning), outperforming solutions from AntGroup and Google AI with a **30x speedup** over the Claude Code baseline.

## Quick Start

### 1. Get API Keys

Each demo uses a different LLM provider. You only need the key(s) for the demo(s) you want to run.

| Key | Demo | Get it at |
|-----|------|-----------|
| `ANTHROPIC_API_KEY` | DABStep inference + General QA (Claude Haiku) | [console.anthropic.com](https://console.anthropic.com) |
| `OPENAI_API_KEY` | EDA notebook gen (GPT-5 mini) | [platform.openai.com](https://platform.openai.com) |

### 2. Run with Docker

```bash
# Build the container
docker build -t data-explorer-agent .

# Run with your API keys (include whichever keys you have)
docker run -it --rm \
    -p 8888:8888 \
    -p 8000:8000 \
    -e ANTHROPIC_API_KEY=sk-ant-your_key \
    -e OPENAI_API_KEY=sk-your_key \
    data-explorer-agent
```

Open http://localhost:8888 in your browser. It will open directly to the **START_HERE** notebook.

### 3. Run on Brev

```bash
brev start https://github.com/PicoNVIDIA/data-explorer-launchable --setup-path setup.sh
```

### 4. Explore the Demo Notebooks

| Notebook | Description | API Key |
|----------|-------------|---------|
| `notebooks/START_HERE.ipynb` | **Start here** -- Overview, results, architecture, and interactive quick demo | `ANTHROPIC_API_KEY` |
| `notebooks/demo_dabstep.ipynb` | **DABStep Inference** -- The leaderboard-winning agent with before/after comparison | `ANTHROPIC_API_KEY` |
| `notebooks/demo_qa.ipynb` | **General Tabular QA** -- Ask questions about any dataset with no prior knowledge | `ANTHROPIC_API_KEY` |
| `notebooks/demo_eda.ipynb` | **Open-ended EDA** -- Upload any tabular data and get an automated analysis notebook | `OPENAI_API_KEY` |

## What's Inside

### DABStep Benchmark Inference

The agent answers complex, multi-step questions about a financial payments dataset. It uses pre-distilled domain knowledge (`helper.py`) and few-shot examples to solve tasks in seconds.

- **LLM**: Claude Haiku via Anthropic API
- **Workflow**: Tool-calling agent with stateful Python executor
- **Output**: Answer + full reasoning trace

### General Tabular QA

Point the agent at any dataset and ask a natural language question. No prior knowledge, no helper functions -- it figures everything out from scratch.

- **LLM**: Claude Haiku via Anthropic API
- **Workflow**: Tool-calling agent with Python executor
- **Output**: Answer with code trace

### Open-Ended Exploratory Data Analysis

Give the agent any CSV, Parquet, or JSON file and an optional prompt. It generates a complete Jupyter notebook with data analysis, statistical summaries, and visualizations -- iteratively refining plots using vision model feedback.

- **LLM**: GPT-5 mini via OpenAI API
- **Workflow**: ReAct agent with notebook manipulation tools
- **Output**: A self-contained Jupyter notebook

### The Three-Phase Approach

1. **Learning** (offline): A heavyweight model solves training tasks and distills reusable functions into `helper.py`
2. **Inference** (fast): A lightweight model uses the distilled code to solve new tasks in seconds
3. **Reflection** (offline): Quality control via self-consistency checks, feeding insights back into inference

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    NVIDIA Launchable                       │
│                                                            │
│  JupyterLab (port 8888)                                   │
│    ├── START_HERE.ipynb     <- start here                 │
│    ├── demo_dabstep.ipynb   -> FastAPI /solve endpoint     │
│    ├── demo_qa.ipynb        -> Generic QA (any dataset)    │
│    └── demo_eda.ipynb       -> EDA notebook generator      │
│                                                            │
│  DABStep Server (port 8000)                                │
│    └── POST /solve -> Tool-calling Agent                   │
│                       └── helper.py (distilled)            │
│                                                            │
│  Data: payments.csv, fees.json, merchant_data.json         │
└────────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
   Anthropic API                    OpenAI API
   (Claude Haiku)                   (GPT-5 mini)
   DABStep + QA demos               EDA demo
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | For DABStep + QA | Anthropic API key from console.anthropic.com |
| `OPENAI_API_KEY` | For EDA | OpenAI API key |
| `JUPYTER_TOKEN` | No | JupyterLab access token (empty = no auth) |

## Run Without Docker

```bash
# Install dependencies
uv sync --all-groups

# Download DABStep data
uv run python dabstep_agent/download_data.py

# Set API keys (include whichever you have)
export ANTHROPIC_API_KEY=sk-ant-your_key
export OPENAI_API_KEY=sk-your_key

# Start the inference server (requires ANTHROPIC_API_KEY)
uv run python dabstep_agent/inference/server.py &

# Start JupyterLab
uv run jupyter lab --port 8888
```

## Team

Built by the KGMON-LLM Agent Research Team at NVIDIA:

- Jiwei Liu (jiweil@nvidia.com)
- Maximilian Jeblick (mjeblick@nvidia.com)
- Jack Yu (jacyu@nvidia.com)
