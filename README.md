# Data Explorer Agent
This project is the NAT implementation of the agentic_EDA module of [CLI-Data-Explorer](https://gitlab-master.nvidia.com/kgmon-llm-tech/cli-data-explorer/-/tree/main/src/cli_data_explorer).

A react agent with tools designed for Jupyter notebook manipulation. It can respond to user questions with data-driven analysis and insights. 
## Features
- **Notebook Manipulation Tools**: Adding/deleting/modifying a Jupyter cell. Automatic notebook execution after each action, and return cell output to the agent.
- **Automated Data Analysis & Visualization**: Generate comprehensive data analysis notebooks with intelligent insights. Add visual plots if needed to complement the notebook report.
- **Vision Analysis Feedback**: Uses a vision language models to provide feedback for generated visualizations so that the agent can iteratively improve the quality of plots inside the notebook. 
- **Automatic Error Fixing**: Automatic notebook debugging and error correction

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

A example Notebook output of the above command is provided in `notebooks/example.ipynb`. 

Use the `--override` option to override values in config file. e.g. `--override workflow.verbose false`; or, change config directly.

## Customize Agent Config
Inside `src/data_explorer_agent/configs/config.yml`, you can customize:
* LLM model
* path to the generated notebook path
* executioni verboseness
* agent instruction (in additional_instructions field)

### Using Your Own Data
A sample QS University Ranking dataset is provided. To use your own data, tell the agent the path to your dataset in input prompt.

### Run Dabstep Eval
```bash
export OPENAI_API_KEY=your_api_key
uv run python dabstep/DABstep_eval.py --task 1
```
