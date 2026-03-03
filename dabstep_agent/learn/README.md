# DABStep Learn

Two-phase pipeline for teaching the DABStep agent to solve tabular data QA tasks: **learn** (discover solutions from scratch) then **distill** (synthesize traces into reusable code).

## Directory Structure

```
learn/
├── learn.py               # Phase 1 – Learning agent
├── learning_config.yml    # NAT workflow config for the learning agent
├── distill_nat/           # Phase 2 – Distill using NAT runtime
│   ├── distill.py
│   └── distill_config.yml
└── distill_agent_sdk/     # Phase 2 – Distill using Claude Agent SDK
    ├── run_distill.py
    └── distill_prompt.md
```

## Phase 1: Learn (`learn.py`)

The learning agent tackles each task from scratch using `bash` (grep) and `python_executor`. It:

1. Reads the question and identifies key terms.
2. Searches `data/context/manual.md` for definitions via grep.
3. Explores data files (CSV/JSON) in `data/context/` with Python.
4. Writes solution code and verifies it against the ground truth.

Traces (full reasoning + generated code) are saved to `dabstep_agent/workspace/learning_traces/`.

### Usage

```bash
# Single task
uv run python dabstep_agent/learn/learn.py --input data/tasks_dev.json --task-id 49

# Multiple tasks
uv run python dabstep_agent/learn/learn.py --input data/tasks_dev.json --task-id 49,50,51

# All tasks (default output: results_learning.jsonl)
uv run python dabstep_agent/learn/learn.py --input data/tasks_dev.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | `data/tasks_dev.json` | Tasks JSON file |
| `--task-id` | all | Comma-separated task IDs to run |
| `--output` | `results_learning.jsonl` | Output JSONL with per-task results |

## Phase 2: Distill

Reads the learning traces and synthesizes them into two artifacts:

- **`helper.py`** — reusable Python module (data loading, fee calculation, field matching, etc.)
- **`solutions.md`** — key insights, function reference, and example solutions for each task

Two interchangeable implementations are provided:

### Option A: NAT runtime (`distill_nat/`)

```bash
uv run python dabstep_agent/learn/distill_nat/distill.py
uv run python dabstep_agent/learn/distill_nat/distill.py --traces-dir dabstep_agent/workspace/learning_traces
```

### Option B: Claude Agent SDK (`distill_agent_sdk/`)

```bash
python dabstep_agent/learn/distill_agent_sdk/run_distill.py
python dabstep_agent/learn/distill_agent_sdk/run_distill.py --traces-dir dabstep_agent/workspace/learning_traces
python dabstep_agent/learn/distill_agent_sdk/run_distill.py --output-dir dabstep_agent/agent_sdk_distill_results
```

## Configuration

Both phases use YAML config files that define:

- **LLM**: Model endpoint, API key (via `$NV_INFER_API_KEY`), and model name.
- **Tools**: `python_executor` (with timeout and workspace dir) and `run_bash`.
- **Workflow**: Max iterations, history length, and system instructions.

See `learning_config.yml` and `distill_nat/distill_config.yml` for details.

## Environment

Requires `$NV_INFER_API_KEY` set in your environment (or `.env` file).
