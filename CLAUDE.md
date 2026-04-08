# Data Explorer Agent - NVIDIA Launchable

## Project Overview

This is an NVIDIA Launchable for the Data Explorer agent, which won **#1 on the DABStep leaderboard** (Data Agent Benchmark for Multi-step Reasoning). Built by the KGMON-LLM Agent Research Team at NVIDIA (Jiwei Liu, Maximilian Jeblick, Jack Yu).

Patrick Moorhead (PicoNVIDIA) built the launchable packaging. The core agent code is from the team's GitLab repo: `gitlab-master.nvidia.com/kgmon-llm-tech/data-explorer-agent` (branch: `generic_qa_agent`).

## Architecture

Docker container running two services:
- **JupyterLab** (port 8888) -- user-facing demo notebooks
- **FastAPI server** (port 8000) -- DABStep inference agent via `/solve` endpoint

All LLMs are accessed via **NVIDIA Inference Hub** (`inference-api.nvidia.com`) using `sk-` API keys. No local GPU needed (CPU-only container).

Built on **NeMo Agent Toolkit (NAT)** which provides the agent loops, tool registration, and LLM abstraction.

## Three Demo Notebooks

### 1. `notebooks/demo_dabstep.ipynb` (main demo)
- **Model**: `nvidia/nvidia/nemotron-3-super-v3` (Nemotron 3 Super)
- **Config**: `dabstep_agent/inference/dabstep_config.yml`
- **Agent type**: `tool_calling_agent` with `python_executor`
- Shows the leaderboard-winning agent answering financial payments questions
- Uses distilled `helper.py` (22 functions) + few-shot examples (`new_solutions.md`)
- **Before/after comparison**: runs 3 hard questions from Jiwei's test set (NOT in training data) with and without the distilled knowledge
- The FastAPI server (`dabstep_agent/inference/server.py`) must be running for this notebook

### 2. `notebooks/demo_qa.ipynb` (general QA)
- **Model**: `aws/anthropic/claude-haiku-4-5-v1` (Claude Haiku)
- **Config**: `generic_qa_agent/config.yml`
- **Agent type**: `tool_calling_agent` with `python_executor`
- Generic QA agent that works on any dataset with no prior knowledge (no helper.py)
- Built by Alessio from the team (from MR #12 / `generic_qa_agent` branch)

### 3. `notebooks/demo_eda.ipynb` (EDA notebook generation)
- **Model**: `openai/openai/gpt-5-mini` (GPT-5 mini)
- **Config**: `src/data_explorer_agent/configs/config.yml`
- **Agent type**: `react_agent` with `notebook_function_group`
- Agent auto-generates a complete Jupyter notebook with analysis and charts from any CSV
- Uses vision analyzer for plot feedback
- **IMPORTANT**: The `react_agent` only works with GPT-5 mini. Nemotron Super and Claude Haiku fail because they output in formats the ReAct parser doesn't understand (think tags, XML). Do NOT change this model without also changing the agent type.

## Key Technical Details

### Models and Endpoints
All models are on NVIDIA Inference Hub (`https://inference-api.nvidia.com`). Requires `sk-` keys (not `nvapi-` keys from build.nvidia.com). VPN required.

| Config | Model | Agent Type |
|--------|-------|------------|
| dabstep_config.yml | `nvidia/nvidia/nemotron-3-super-v3` | tool_calling_agent |
| generic_qa_agent/config.yml | `aws/anthropic/claude-haiku-4-5-v1` | tool_calling_agent |
| config.yml (EDA) | `openai/openai/gpt-5-mini` | react_agent |

Nemotron 3 Super requires `temperature: 0.7` in the config or it returns empty responses.

### The 3-Phase DABStep Architecture
1. **Learning** (offline): Heavy model (Opus) solves training tasks, distills reusable functions into `helper.py` and few-shot examples (`new_solutions.md`)
2. **Inference** (fast): Lightweight model uses distilled code to answer questions in seconds
3. **Reflection** (offline): Quality checks feed insights back into inference prompts

The launchable only demonstrates Phase 2 (inference). Learning and distillation are excluded (too time-consuming, NAT recursion bug with distill).

### Python Executor Fixes
`src/data_explorer_agent/python_executor.py` has two fixes applied:
- **Newline unescaping**: Some models return `\\n` instead of real newlines in tool call code. The executor detects and fixes this.
- **Repetition detector**: If the same code is submitted 2+ times, returns a STOP message telling the model to return its final answer instead of looping.

### Comparison Questions
The before/after comparison in `demo_dabstep.ipynb` uses questions from Jiwei's test set (task IDs 1712, 1810, 2644). These are NOT in the training data or `new_solutions.md`, so the comparison is fair. The "from scratch" agent sometimes hits context window limits (131K tokens) on complex questions -- this is handled gracefully and actually demonstrates why the learning phase matters.

### File Structures Cache
`dabstep_agent/generate_file_structures.py` creates `data/context/file_structures.json` at Docker build time. This gives the agent schema information about the data files without having to explore them. The Dockerfile runs this after downloading the DABStep data from HuggingFace.

## Docker Setup

```bash
docker build -t data-explorer-agent .
docker run -d --name dea-test \
    -p 8888:8888 -p 8000:8000 \
    -e NV_INFER_API_KEY=your_sk_key_here \
    data-explorer-agent
```

The `entrypoint.sh` starts the FastAPI server in the background, then launches JupyterLab.

## GitLab
Team's internal GitLab repo: `gitlab-master.nvidia.com/kgmon-llm-tech/data-explorer-agent`. Requires a GitLab personal access token to pull.

## GitHub Repo
Public repo: `https://github.com/PicoNVIDIA/data-explorer-launchable`

## Team Branches on GitLab
- `generic_qa_agent` -- our base branch (Alessio's generic QA agent)
- `EDA_example` -- Jack's EDA cleanup with multi-turn chat loop (not fully integrated)
- `dabstep_agent_frontier` -- original DABStep code
- `aledev/kdd` -- Alessio's KDD work

## Known Issues
- EDA demo only works with GPT-5 mini via Inference Hub. Other models fail the ReAct parser.
- Nemotron 3 Super gets ~77% accuracy on hard DABStep questions (vs 89.95% with Claude Haiku). Some answers may be slightly off.
- The "from scratch" comparison can hit context window limits on complex questions (expected behavior).
- Inference Hub requires NVIDIA VPN connection.
- Brev deployment not yet done -- waiting on review.

## People
- **Patrick Moorhead** (pmoorhead@nvidia.com) -- built the launchable
- **Jiwei Liu** (jiweil@nvidia.com) -- team lead, DABStep architecture
- **Maximilian Jeblick** (mjeblick@nvidia.com) -- Nemotron Super testing
- **Jack Yu** (jacyu@nvidia.com) -- EDA workflow, multi-turn chat
- **Alessio** -- generic QA agent
- **Jean-Francois Puget** (jpuget@nvidia.com) -- manager
- **Bartley Richardson** (brichardson@nvidia.com) -- requested the launchable
- **Brad Nemire** (bnemire@nvidia.com) -- Brev/launchable team
