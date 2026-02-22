"""
REST API for DABStep agent.

Usage:
    uv run python dabstep_agent/api.py
    uv run python dabstep_agent/api.py --port 8080 --host 0.0.0.0

Request:
    POST /solve
    {"question": "...", "guidelines": "..."}

Response:
    {"agent_answer": "...", "reasoning_trace": [...]}
"""

import asyncio
import argparse
import json
import os
import sys
import time

from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

DIR = os.path.dirname(os.path.abspath(__file__))

# Add dabstep_agent dir to sys.path so we can import solve.py directly
if DIR not in sys.path:
    sys.path.insert(0, DIR)

import solve as _solve_mod
from solve import (
    CONFIG,
    DATA_DIR,
    extract_agent_answer,
    load_file_structures,
    build_prompt,
)
from postprocess import extract_nested_answer, round_if_delta_question
from data_explorer_agent.python_executor import _tools as executor_tools
from nat.runtime.loader import load_workflow

# ---------------------------------------------------------------------------
# Globals initialised at startup
# ---------------------------------------------------------------------------
_workflow = None
_file_structures = None
_examples = None


@asynccontextmanager
async def lifespan(app):
    """Start up: load workflow, file structures, and examples."""
    global _workflow, _file_structures, _examples

    _file_structures = load_file_structures()

    examples_path = os.path.join(DIR, "new_solutions.md")
    with open(examples_path) as f:
        _examples = f.read()

    async with load_workflow(CONFIG) as wf:
        _workflow = wf
        print("Workflow loaded — API ready.")
        yield

    _workflow = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="DABStep Agent API", lifespan=lifespan)


class SolveRequest(BaseModel):
    question: str
    guidelines: str = "N/A"


class SolveResponse(BaseModel):
    agent_answer: str
    reasoning_trace: list


# Use a lock so tasks don't interleave (shared python executor state)
_solve_lock = asyncio.Lock()


SOLVE_TIMEOUT = int(os.environ.get("SOLVE_TIMEOUT", 270))  # seconds
SOLVE_LOG = os.environ.get("SOLVE_LOG", os.path.join(DIR, "solve_log.jsonl"))


def _append_log(record: dict):
    with open(SOLVE_LOG, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.post("/solve", response_model=SolveResponse)
async def solve(req: SolveRequest):
    async with _solve_lock:
        t0 = time.time()
        question = {"question": req.question, "guidelines": req.guidelines}
        prompt = build_prompt(question, _file_structures, _examples)

        try:
            async with _workflow.run(prompt) as runner:
                raw_answer = await asyncio.wait_for(
                    runner.result(), timeout=SOLVE_TIMEOUT
                )
        except asyncio.TimeoutError:
            elapsed = round(time.time() - t0, 2)
            print(f"Solve timed out after {SOLVE_TIMEOUT}s")
            # Reset executor state before returning
            try:
                await executor_tools["reset_environment"]()
            except Exception:
                pass
            _append_log({
                "question": req.question,
                "guidelines": req.guidelines,
                "agent_answer": "",
                "reasoning_trace": [],
                "time_seconds": elapsed,
            })
            return SolveResponse(agent_answer="", reasoning_trace=[])
        except Exception as e:
            elapsed = round(time.time() - t0, 2)
            trace = list(_solve_mod._last_messages)
            _append_log({
                "question": req.question,
                "guidelines": req.guidelines,
                "agent_answer": f"ERROR: {e}",
                "reasoning_trace": trace,
                "time_seconds": elapsed,
            })
            return SolveResponse(
                agent_answer=f"ERROR: {e}",
                reasoning_trace=trace,
            )

        agent_answer = extract_agent_answer(raw_answer)
        agent_answer = extract_nested_answer(agent_answer)
        agent_answer = round_if_delta_question(req.question, agent_answer)
        agent_answer = str(agent_answer)
        trace = list(_solve_mod._last_messages)
        elapsed = round(time.time() - t0, 2)

        # Reset executor state for next request
        try:
            await executor_tools["reset_environment"]()
        except Exception:
            pass

        _append_log({
            "question": req.question,
            "guidelines": req.guidelines,
            "agent_answer": agent_answer,
            "reasoning_trace": trace,
            "time_seconds": elapsed,
        })

        return SolveResponse(agent_answer=agent_answer, reasoning_trace=trace)


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="DABStep Agent API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)
