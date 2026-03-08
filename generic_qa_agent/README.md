# Generic Dataset QA Agent

A generic question-answering pipeline for tabular data. Unlike the DABStep inference pipeline, this agent:

- Works with **any dataset folder** (CSV, JSON, JSONL, Parquet, etc.)
- Accepts **any natural-language question**
- Uses **no prior knowledge** (no helper.py, no domain-specific examples)
- Is not tied to any specific dataset format or schema

The agent explores the data directory autonomously, then writes Python code to answer the question.

## Usage

```bash
uv run python generic_qa_agent/solve.py \
    --data-dir data_folder_path \
    --question "your question here"
```

For example
```bash
uv run python generic_qa_agent/solve.py \
    --data-dir  data/context \
    --question "which country has the most fraud?"
```

## Configuration

Edit `generic_qa_agent/config.yml` to change the LLM model or agent parameters.
