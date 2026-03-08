# Data Explorer Agent

This project is the NAT implementation of the agentic_EDA module of CLI-Data-Explorer.

A react agent with tools designed for Jupyter notebook manipulation. It can respond to user questions with data-driven analysis and insights.

Using Data Explorer Agent, we achieved **1st place** on the [DABStep](https://huggingface.co/datasets/adyen/DABstep) tabular data QA benchmark. See [dabstep_agent/](dabstep_agent/) for the full pipeline.

## Features
- **Notebook Manipulation Tools**: Adding/deleting/modifying a Jupyter cell. Automatic notebook execution after each action, and return cell output to the agent.
- **Automated Data Analysis & Visualization**: Generate comprehensive data analysis notebooks with intelligent insights. Add visual plots if needed to complement the notebook report.
- **Vision Analysis Feedback**: Uses vision language models to provide feedback for generated visualizations so that the agent can iteratively improve the quality of plots inside the notebook.
- **Automatic Error Fixing**: Automatic notebook debugging and error correction
- **Bash Tool Support**: Safe bash command execution with read-only commands (ls, find, grep, cat, head, tail, wc, pwd, echo, file, stat, du, df). Destructive commands are blocked for safety.

## Installation & Setup

```bash
# install `uv` if you haven't already. Then:
cd data_explorer_agent
uv sync

export OPENAI_API_KEY="your-api-key"
```

## Quick Start

The agent is NAT compatible. There are two ways to invoke the agent.

1. From python file, reference code in example.py
    ```bash
    uv run python example.py
    ```

2. Use NAT in CLI:
    ```bash
    nat run --config_file src/data_explorer_agent/configs/config.yml --input "Based on this dataset in sample_data/QS_2025.csv, report the region that have the strongest education system. Make it a nice Notebook report."
    ```

An example Notebook output of the above command is provided in `notebooks/example.ipynb`.

Use the `--override` option to override values in config file. e.g. `--override workflow.verbose false`; or, change config directly.

## Customize Agent Config
Inside `src/data_explorer_agent/configs/config.yml`, you can customize:
* LLM model
* path to the generated notebook path
* execution verboseness
* agent instruction (in additional_instructions field)

### Alternative Agent Configs
- **bash_tool_call_agent.yml**: A simplified tool calling agent with only bash tools. Useful for file navigation and lookup.
    ```bash
    nat run --config_file src/data_explorer_agent/configs/bash_tool_call_agent.yml --input "List all Python files in the current directory"
    ```

### Using Your Own Data
A sample QS University Ranking dataset is provided. To use your own data, tell the agent the path to your dataset in input prompt.

## DABStep Agent

Agent pipeline that won **1st place** on the [DABStep benchmark](https://huggingface.co/datasets/adyen/DABstep). The workflow has three stages: **download data**, **learn** (discover solutions + distill into reusable code), and **inference** (solve tasks at scale).

All DABStep commands should be run from the root of the repository.

```bash
# 1. Download data
uv run python dabstep_agent/download_data.py

# 2. Learn solutions from scratch, then distill
uv run python dabstep_agent/learn/learn.py --input data/tasks_dev.json --task-id 49,50,51
uv run python dabstep_agent/learn/distill_nat/distill.py          # Option A: distill via NAT
python dabstep_agent/learn/distill_agent_sdk/run_distill.py       # Option B: distill via Claude Agent SDK

# 3. Inference
uv run python dabstep_agent/inference/server.py
uv run python dabstep_agent/inference/client.py --input data/tasks.json
```

See [`dabstep_agent/README.md`](dabstep_agent/README.md) for full details.
