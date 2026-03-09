"""
Generic dataset QA solver using NAT tool_calling_agent + python_executor.

Runs without any prior knowledge (no helper.py, no domain examples).
Users specify a data directory and ask any question.

Usage:
    uv run python generic_qa_agent/solve.py --data-dir sample_data/ --question "Which university ranked first?"
"""

import argparse
import asyncio
import json
import os
import re
import glob as glob_module

from nat.runtime.loader import load_workflow
from data_explorer_agent.python_executor import _tools as executor_tools
from dotenv import load_dotenv

load_dotenv()

DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(DIR, "config.yml")


# ---------------------------------------------------------------------------
# Data directory scanning
# ---------------------------------------------------------------------------

def scan_data_dir(data_dir: str) -> str:
    """Scan a data directory and return a summary of available files."""
    data_dir = os.path.abspath(data_dir)
    lines = [f"Data directory: {data_dir}"]

    patterns = ["*.csv", "*.json", "*.jsonl", "*.parquet", "*.tsv", "*.xlsx"]
    found_files = []
    for pattern in patterns:
        found_files.extend(glob_module.glob(os.path.join(data_dir, "**", pattern), recursive=True))

    if not found_files:
        lines.append("  (no data files found — agent will explore the directory)")
        return "\n".join(lines)

    for filepath in sorted(found_files):
        filename = os.path.relpath(filepath, data_dir)
        size_kb = os.path.getsize(filepath) / 1024
        ext = os.path.splitext(filepath)[1].lower()

        desc = ""
        try:
            if ext == ".csv":
                import csv
                with open(filepath, newline="", encoding="utf-8", errors="replace") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    desc = f"CSV, columns: {', '.join(header[:8])}"
                    if len(header) > 8:
                        desc += f" (+{len(header)-8} more)"
            elif ext in (".json", ".jsonl"):
                with open(filepath, encoding="utf-8", errors="replace") as f:
                    first_line = f.readline().strip()
                try:
                    obj = json.loads(first_line)
                    if isinstance(obj, dict):
                        keys = list(obj.keys())[:6]
                        desc = f"JSON/JSONL, keys: {', '.join(keys)}"
                    elif isinstance(obj, list):
                        desc = f"JSON array, {len(obj)} items"
                    else:
                        desc = "JSON"
                except json.JSONDecodeError:
                    with open(filepath, encoding="utf-8", errors="replace") as f:
                        f.read(4096)
                    desc = "JSON (see file for structure)"
            else:
                desc = ext.lstrip(".").upper()
        except Exception:
            desc = "unreadable"

        lines.append(f"  - {filename} ({size_kb:.1f} KB) — {desc}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(raw: str) -> str:
    """Extract the answer value from the agent response."""
    if raw is None:
        return ""
    raw = str(raw).strip()

    # Try {"answer": "..."} pattern
    match = re.search(r'\{\s*"answer"\s*:\s*"([^"]*)"\s*\}', raw)
    if match:
        return match.group(1).strip()

    # Try {"answer": <value>} with non-string value
    match = re.search(r'\{\s*"answer"\s*:\s*([^,}\]]+)', raw)
    if match:
        val = match.group(1).strip().strip('"')
        return val

    # Try to parse as JSON
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for key in ("answer", "agent_answer", "result"):
                if key in parsed:
                    return str(parsed[key]).strip()
    except (json.JSONDecodeError, TypeError):
        pass

    return raw


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(question: str, data_dir: str, file_summary: str) -> str:
    abs_data_dir = os.path.abspath(data_dir)
    return (
        f"Available data files:\n{file_summary}\n"
        f"\nData directory absolute path: {abs_data_dir}\n"
        f"\nQUESTION: {question}"
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

async def solve_question(workflow, question: str, data_dir: str,
                         file_summary: str) -> str:
    """Run one question through the NAT workflow and return the raw answer."""
    prompt = build_prompt(question, data_dir, file_summary)
    async with workflow.run(prompt) as runner:
        return await runner.result()


async def main(args):
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")

    file_summary = scan_data_dir(data_dir)
    print(f"\n{file_summary}\n")

    question = args.question
    print(f"Question: {question}\n")

    async with load_workflow(CONFIG) as workflow:
        try:
            raw_answer = await solve_question(workflow, question, data_dir, file_summary)
            answer = extract_answer(raw_answer)
        except Exception as e:
            answer = ""
            print(f"ERROR: {e}")

        # Save the generated code to solution.py
        try:
            await executor_tools["save_generated_code"]("solution")
            src = os.path.join(DIR, "workspace", "tasksolution.py")
            dst = os.path.join(DIR, "solution.py")
            if os.path.exists(src):
                os.rename(src, dst)
                print(f"Solution code saved to: {dst}")
        except Exception:
            pass

    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generic dataset QA solver (no prior knowledge)",
    )
    parser.add_argument("--data-dir", required=True,
                        help="Path to the folder containing data files")
    parser.add_argument("--question", required=True,
                        help="The question to answer about the data")
    args = parser.parse_args()

    asyncio.run(main(args))
