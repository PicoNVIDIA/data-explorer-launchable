"""
DABStep learning phase: discover solutions from scratch using bash + python_executor.

The agent uses bash (grep) to search manual.md for term definitions,
then explores data with python_executor, iterating until it produces
code that matches the ground truth.

Usage:
    uv run python dabstep_agent/learn/learn.py --input data/tasks_dev.json --task-id 49
    uv run python dabstep_agent/learn/learn.py --input data/tasks_dev.json --task-id 49,50,51
"""

import argparse
import asyncio
import json
import os
import re
import time

import yaml
from nat.runtime.loader import load_workflow
from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraph
from data_explorer_agent.python_executor import _tools as executor_tools
from dotenv import load_dotenv

load_dotenv()

DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(DIR, "learning_config.yml")
DATA_DIR = "data/context"

with open(CONFIG) as _f:
    _config = yaml.safe_load(_f)
WORKSPACE_DIR = _config["function_groups"]["python_executor"]["workspace_dir"]

# ---------------------------------------------------------------------------
# Monkey-patch agent_node to capture full LLM message history (same as solve.py)
# ---------------------------------------------------------------------------
_last_messages = []
_original_agent_node = ToolCallAgentGraph.agent_node


def _serialize_messages(messages) -> list[dict]:
    out = []
    for m in messages:
        entry = {"role": m.type, "content": m.content}
        if hasattr(m, "name") and m.name:
            entry["tool_name"] = m.name
        if hasattr(m, "tool_calls") and m.tool_calls:
            entry["tool_calls"] = m.tool_calls
        if hasattr(m, "tool_call_id") and m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id
        out.append(entry)
    return out


async def _tracing_agent_node(self, state):
    global _last_messages
    result = await _original_agent_node(self, state)
    _last_messages = _serialize_messages(result.messages)
    return result


ToolCallAgentGraph.agent_node = _tracing_agent_node


def save_trace(
    task_id: str,
    question: dict,
    agent_answer: str,
    match: bool | None,
    elapsed: float,
    trace_dir: str = WORKSPACE_DIR,
):
    os.makedirs(trace_dir, exist_ok=True)
    record = {
        "task_id": task_id,
        "question": question["question"],
        "guidelines": question.get("guidelines", "N/A"),
        "ground_truth": question.get("answer", ""),
        "agent_answer": agent_answer,
        "match": match,
        "time_seconds": elapsed,
        "reasoning_trace": list(_last_messages),
    }
    path = os.path.join(trace_dir, f"{task_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False, default=str)
    print(f"Trace saved: {path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_file_structures(data_dir: str = DATA_DIR) -> str:
    cache_path = os.path.join(data_dir, "file_structures.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            structures = json.load(f)
        lines = []
        for filename, structure in structures.items():
            if "error" in structure:
                lines.append(f"- {filename}: (error extracting structure)")
                continue
            file_type = structure.get("file_type", "unknown")
            if file_type == "csv":
                sample = structure.get("sample_row", {})
                lines.append(f"- {filename} (CSV):")
                lines.append(f"    Sample row: {json.dumps(sample, default=str)}")
            elif file_type == "json":
                keys = structure.get("keys", [])
                structure_type = structure.get("structure_type", "unknown")
                sample = structure.get("sample_record", {})
                lines.append(f"- {filename} (JSON, {structure_type}):")
                lines.append(f"    Keys: {', '.join(keys)}")
                lines.append(f"    Sample record: {json.dumps(sample, default=str)}")
        return "\n".join(lines)
    return f"Data directory: {data_dir}/ (scan for csv/json/jsonl files)"


def extract_agent_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = str(answer).strip()
    # Match both quoted string values and unquoted numeric values
    json_match = re.search(r'\{\s*"agent_answer"\s*:\s*"([^"]*)"\s*\}', answer)
    if json_match:
        return json_match.group(1).strip()
    json_match = re.search(r'\{\s*"agent_answer"\s*:\s*([-\d.eE+]+)\s*\}', answer)
    if json_match:
        return json_match.group(1).strip()
    # Try parsing the whole string as JSON
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, dict) and "agent_answer" in parsed:
            return str(parsed["agent_answer"]).strip()
    except (json.JSONDecodeError, TypeError):
        pass
    # Try to find a JSON object embedded in larger text
    json_obj_match = re.search(r'\{[^{}]*"agent_answer"[^{}]*\}', answer)
    if json_obj_match:
        try:
            parsed = json.loads(json_obj_match.group(0))
            if "agent_answer" in parsed:
                return str(parsed["agent_answer"]).strip()
        except (json.JSONDecodeError, TypeError):
            pass
    return answer


def normalize_answer(answer: str) -> str:
    answer = extract_agent_answer(answer)
    try:
        parts = [p.strip() for p in answer.split(",")]
        numbers = [float(p) for p in parts if p]
        if all(n == int(n) for n in numbers):
            numbers = [int(n) for n in numbers]
        numbers.sort()
        return ", ".join(str(n) for n in numbers)
    except (ValueError, AttributeError):
        return answer


def compare_answers(agent_answer: str, ground_truth: str) -> bool:
    return normalize_answer(agent_answer) == normalize_answer(ground_truth)


# ---------------------------------------------------------------------------
# Prompt builder — learning mode (no examples, no helper.py)
# ---------------------------------------------------------------------------

def build_learning_prompt(question: dict, file_structures: str) -> str:
    gt = question.get("answer", "")
    return f"""You are solving a data analysis question. You must figure out the answer from scratch.

Available data files in '{DATA_DIR}/':
{file_structures}

There is also a detailed manual at '{DATA_DIR}/manual.md' that defines key terms and concepts.

IMPORTANT: Follow this step-by-step approach:

STEP 1 - UNDERSTAND THE QUESTION
Read the question carefully. Identify key terms that need precise definitions (e.g. "fraud", "fee", "intracountry", "ACI", account types, etc.)

STEP 2 - SEARCH THE MANUAL
Use run_bash to grep for definitions of key terms in the manual:
  grep -i "fraud" data/context/manual.md
  grep -i -A 5 "some_term" data/context/manual.md

STEP 3 - EXPLORE THE DATA
Use execute_python_code to load and inspect the relevant data files. Check column names, data types, unique values, and distributions.

STEP 4 - WRITE SOLUTION CODE
Based on the manual definitions and your data exploration, write Python code that computes the answer.

STEP 5 - VERIFY
The expected answer is: {gt}
Run your code, check that it matches. If not, re-read relevant parts of the manual and fix your logic.

QUESTION: {question['question']}

GUIDELINES: {question.get('guidelines', 'N/A')}

EXPECTED ANSWER: {gt}
"""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

async def solve_task(workflow, question: dict, file_structures: str) -> str:
    prompt = build_learning_prompt(question, file_structures)
    async with workflow.run(prompt) as runner:
        return await runner.result()


async def main(args):
    with open(args.input) as f:
        tasks = json.load(f)

    file_structures = load_file_structures()

    if args.task_id:
        filter_ids = set(args.task_id.split(","))
        task_list = [(i, t) for i, t in enumerate(tasks) if t.get("task_id") in filter_ids]
    else:
        task_list = list(enumerate(tasks))

    output_file = args.output or "results_learning.jsonl"
    open(output_file, "w").close()

    print(f"\n{'='*70}")
    print(f"Learning phase: {len(task_list)} tasks -> {output_file}")
    print(f"{'='*70}")

    correct = 0
    total = 0

    async with load_workflow(CONFIG) as workflow:
        for idx, question in task_list:
            task_id = question.get("task_id", str(idx))
            gt = question.get("answer", "")

            print(f"\n--- Task {task_id} ---")
            print(f"Q: {question['question'][:120]}...")
            print(f"Expected: {gt}")

            t0 = time.time()
            try:
                raw_answer = await solve_task(workflow, question, file_structures)
                extracted = extract_agent_answer(raw_answer)
                match = compare_answers(raw_answer, gt) if gt else None
            except Exception as e:
                raw_answer = ""
                extracted = ""
                match = False
                print(f"ERROR: {e}")

            elapsed = round(time.time() - t0, 2)

            save_trace(task_id, question, extracted, match, elapsed)

            try:
                await executor_tools["save_generated_code"](task_id)
                await executor_tools["reset_environment"]()
            except Exception:
                pass

            total += 1
            if match:
                correct += 1

            print(f"Agent answer: {extracted}")
            print(f"Ground truth: {gt}")
            print(f"Match: {match}  ({elapsed}s)")
            print(f"Running accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

            record = {
                "task_id": task_id,
                "agent_answer": extracted,
                "ground_truth": gt,
                "match": match,
                "time_seconds": elapsed,
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")

    print(f"\nLearning phase done. {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"Results: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DABStep learning phase")
    parser.add_argument("--input", default="data/tasks_dev.json", help="Tasks JSON file")
    parser.add_argument("--task-id", default=None, help="Comma-separated task IDs")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    args = parser.parse_args()

    asyncio.run(main(args))
