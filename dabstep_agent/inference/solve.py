"""
DABStep task solver using NAT tool_calling_agent + python_executor.

Replaces: run_all_tasks.sh + run.py + simple_qa_agent.py

Usage:
    uv run python dabstep_agent/inference/solve.py --input data/tasks_dev.json
    uv run python dabstep_agent/inference/solve.py --input data/tasks.json --task-id 1433,1434
    uv run python dabstep_agent/inference/solve.py --input data/tasks.json --passes 3
"""

import argparse
import asyncio
import json
import os
import re

from nat.runtime.loader import load_workflow
from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraph
from data_explorer_agent.python_executor import _tools as executor_tools
from dotenv import load_dotenv

load_dotenv()

DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(DIR, "dabstep_config.yml")
DATA_DIR = "data/context"

# ---------------------------------------------------------------------------
# Monkey-patch agent_node to capture the full LLM message history.
# state.messages grows each iteration; we snapshot it after every LLM call.
# The last snapshot is the complete conversation.
# ---------------------------------------------------------------------------
_last_messages = []  # module-level; overwritten each agent_node call
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


def save_trace(task_id: str, trace_dir: str = os.path.join(DIR, "workspace/traces")):
    """Write the captured message history to traces/<task_id>.json."""
    os.makedirs(trace_dir, exist_ok=True)
    path = os.path.join(trace_dir, f"{task_id}.json")
    with open(path, "w") as f:
        json.dump(_last_messages, f, indent=2, ensure_ascii=False, default=str)
    print(f"Trace saved: {path}")


# ---------------------------------------------------------------------------
# Helpers ported from dabstep_agent/run.py
# ---------------------------------------------------------------------------

def load_file_structures(data_dir: str = DATA_DIR) -> str:
    """Load cached file structures, or return fallback."""
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
    """Extract the agent_answer value from a response."""
    if answer is None:
        return ""
    answer = str(answer).strip()
    json_match = re.search(r'\{\s*"agent_answer"\s*:\s*"([^"]*)"\s*\}', answer)
    if json_match:
        return json_match.group(1).strip()
    try:
        parsed = json.loads(answer)
        if isinstance(parsed, dict) and "agent_answer" in parsed:
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
# Prompt builder (from simple_qa_agent.py)
# ---------------------------------------------------------------------------

def build_prompt(question: dict, file_structures: str) -> str:
    return f"""
Available data files in '{DATA_DIR}/':
{file_structures}

You have access to the data files listed above. Use Python (pandas, numpy, etc.) via the python_executor tool to load, explore, and analyze the data as needed. Read any documentation files (e.g. manual.md, *-readme.md) in '{DATA_DIR}/' first to understand the data semantics before answering.

QUESTION: {question['question']}

GUIDELINES: {question.get('guidelines', 'N/A')}
"""


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

async def solve_task(workflow, question: dict, file_structures: str) -> str:
    """Send one task through the NAT workflow and return the raw answer."""
    prompt = build_prompt(question, file_structures)
    async with workflow.run(prompt) as runner:
        return await runner.result()


async def main(args):
    # Load inputs
    with open(args.input) as f:
        tasks = json.load(f)

    file_structures = load_file_structures()

    # Filter tasks if --task-id provided
    if args.task_id:
        filter_ids = set(args.task_id.split(","))
        task_list = [(i, t) for i, t in enumerate(tasks) if t.get("task_id") in filter_ids]
    else:
        task_list = list(enumerate(tasks))

    for pass_num in range(1, args.passes + 1):
        output_file = args.output or f"results_nat_pass{pass_num}.jsonl"
        if args.passes > 1:
            base, ext = os.path.splitext(output_file)
            output_file = f"{base}_pass{pass_num}{ext}"

        # Clear output file
        open(output_file, "w").close()

        print(f"\n{'='*70}")
        print(f"Pass {pass_num}/{args.passes} -- {len(task_list)} tasks -> {output_file}")
        print(f"{'='*70}")

        correct = 0
        total = 0

        async with load_workflow(CONFIG) as workflow:
            for idx, question in task_list:
                task_id = question.get("task_id", str(idx))
                gt = question.get("answer", "")

                print(f"\n--- Task {task_id} ---")
                print(f"Q: {question['question'][:120]}...")

                try:
                    raw_answer = await solve_task(workflow, question, file_structures)
                    extracted = extract_agent_answer(raw_answer)
                    match = compare_answers(raw_answer, gt) if gt else None
                except Exception as e:
                    raw_answer = ""
                    extracted = ""
                    match = False
                    print(f"ERROR: {e}")

                save_trace(task_id)

                # Save generated code and reset — called directly, no LLM needed.
                try:
                    await executor_tools["save_generated_code"](task_id)
                    await executor_tools["reset_environment"]()
                except Exception:
                    pass

                total += 1
                if match:
                    correct += 1

                print(f"Agent answer: {extracted}")
                if gt:
                    print(f"Ground truth: {gt}")
                    print(f"Match: {match}")
                print(f"Running accuracy: {correct}/{total} ({correct/total*100:.1f}%)")

                record = {"task_id": task_id, "agent_answer": extracted}
                with open(output_file, "a") as f:
                    f.write(json.dumps(record) + "\n")


        print(f"\nPass {pass_num} done. {correct}/{total} correct ({correct/total*100:.1f}%)")
        print(f"Results: {output_file}")

    print(f"\nAll {args.passes} passes completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DABStep solver (NAT)")
    parser.add_argument("--input", default="data/tasks.json", help="Tasks JSON file")
    parser.add_argument("--task-id", default=None, help="Comma-separated task IDs (optional)")
    parser.add_argument("--passes", type=int, default=1, help="Number of passes")
    parser.add_argument("--output", default=None, help="Output JSONL file")
    args = parser.parse_args()

    asyncio.run(main(args))
