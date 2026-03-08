# DABStep Inference

REST API and CLI for solving DABStep fee-calculation tasks using NAT tool_calling_agent + python_executor.

## Files

| File | Description |
|---|---|
| `server.py` | FastAPI server exposing a `/solve` endpoint |
| `client.py` | HTTP client that sends tasks to the server |
| `solve.py` | Core solver: prompt building, answer extraction, tracing |
| `postprocess.py` | Post-processing for JSONL answer files (nested answer extraction, rounding) |
| `dabstep_config.yml` | NAT workflow config (LLM, tools, system prompt) |
| `new_solutions.md` | Few-shot examples injected into the solver prompt |

## Usage

### Run the API server

```bash
uv run python dabstep_agent/inference/server.py
uv run python dabstep_agent/inference/server.py --port 8080 --host 0.0.0.0
```

### Send tasks via the client

```bash
uv run python dabstep_agent/inference/client.py --input data/tasks.json
uv run python dabstep_agent/inference/client.py --input data/tasks.json --task-id 1712,1810
```

### Run the solver directly (no server)

```bash
uv run python dabstep_agent/inference/solve.py --input data/tasks.json
uv run python dabstep_agent/inference/solve.py --input data/tasks.json --task-id 1433,1434 --passes 3
```

### Post-process results

```bash
uv run python dabstep_agent/inference/postprocess.py results.jsonl results_clean.jsonl
```
