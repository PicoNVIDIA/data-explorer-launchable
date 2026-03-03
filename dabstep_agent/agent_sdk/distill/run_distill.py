"""
DABStep distill phase using Claude Agent SDK.

Reads learning traces (*.json + *.py) and prompts Claude to synthesize
them into a clean helper.py + solutions.md.

Usage:
    python dabstep_agent/agent_sdk/distill/run_distill.py
    python dabstep_agent/agent_sdk/distill/run_distill.py --traces-dir dabstep_agent/workspace/learning_traces
    python dabstep_agent/agent_sdk/distill/run_distill.py --output-dir dabstep_agent
"""

import argparse
import asyncio
import glob
import json
import os
import time

from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DIR, "..", "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "context")
DEFAULT_TRACES_DIR = os.path.join(PROJECT_ROOT, "dabstep_agent", "workspace", "learning_traces")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dabstep_agent", "agent_sdk_distill_results")
PROMPT_TEMPLATE = os.path.join(DIR, "distill_prompt.md")


# ---------------------------------------------------------------------------
# Collect learning traces
# ---------------------------------------------------------------------------

def load_traces(traces_dir: str) -> list[dict]:
    """Load all .json trace files and pair with .py code files."""
    traces = []
    for json_path in sorted(glob.glob(os.path.join(traces_dir, "*.json"))):
        with open(json_path) as f:
            trace = json.load(f)

        task_id = trace.get("task_id", os.path.basename(json_path).replace(".json", ""))
        py_path = os.path.join(traces_dir, f"task{task_id}.py")
        code = ""
        if os.path.exists(py_path):
            with open(py_path) as f:
                code = f.read()

        traces.append({
            "task_id": task_id,
            "question": trace.get("question", ""),
            "guidelines": trace.get("guidelines", ""),
            "ground_truth": trace.get("ground_truth", ""),
            "agent_answer": trace.get("agent_answer", ""),
            "match": trace.get("match", False),
            "code": code,
        })
    return traces


def load_file_structures(data_dir: str) -> str:
    """Load cached file structures for context."""
    cache_path = os.path.join(data_dir, "file_structures.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            structures = json.load(f)
        lines = []
        for filename, structure in structures.items():
            if "error" in structure:
                continue
            file_type = structure.get("file_type", "unknown")
            if file_type == "csv":
                sample = structure.get("sample_row", {})
                cols = structure.get("columns", [])
                lines.append(f"- {filename} (CSV):")
                lines.append(f"    Columns: {', '.join(cols)}")
                lines.append(f"    Sample: {json.dumps(sample, default=str)}")
            elif file_type == "json":
                keys = structure.get("keys", [])
                sample = structure.get("sample_record", {})
                lines.append(f"- {filename} (JSON, {structure.get('structure_type', '')}):")
                lines.append(f"    Keys: {', '.join(keys)}")
                lines.append(f"    Sample: {json.dumps(sample, default=str)}")
        return "\n".join(lines)
    return f"Data directory: {data_dir}/ (scan for csv/json files)"


def format_trace_summary(trace: dict) -> str:
    """Format a single trace for inclusion in the prompt."""
    status = "CORRECT" if trace["match"] else "FAILED"
    code_snippet = trace["code"][:3000] if trace["code"] else "(no code saved)"

    return f"""### Task {trace['task_id']} [{status}]
Question: {trace['question']}
Guidelines: {trace['guidelines']}
Ground Truth: {trace['ground_truth']}
Agent Answer: {trace['agent_answer']}

Code:
```python
{code_snippet}
```
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(traces: list[dict], data_dir: str, traces_dir: str, output_dir: str) -> str:
    """Build the distill prompt from the template + traces."""
    file_structures = load_file_structures(data_dir)
    traces_text = "\n---\n".join(format_trace_summary(t) for t in traces)

    with open(PROMPT_TEMPLATE) as f:
        template = f.read()

    return template.format(
        traces_dir=traces_dir,
        data_dir=data_dir,
        file_structures=file_structures,
        traces_text=traces_text,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    traces_dir = os.path.abspath(args.traces_dir)
    output_dir = os.path.abspath(args.output_dir)
    data_dir = os.path.abspath(args.data_dir)

    print(f"Loading traces from: {traces_dir}")
    traces = load_traces(traces_dir)
    print(f"Found {len(traces)} traces: {[t['task_id'] for t in traces]}")
    print(f"  Correct: {sum(1 for t in traces if t['match'])}")
    print(f"  Failed:  {sum(1 for t in traces if not t['match'])}")

    prompt = build_prompt(traces, data_dir, traces_dir, output_dir)
    print(f"\nOutput dir: {output_dir}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"\n{'='*70}")
    print("Starting distill agent (Claude Agent SDK)...")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Grep", "Glob"],
        model=args.model,
        max_turns=args.max_turns,
        cwd=PROJECT_ROOT,
        permission_mode="bypassPermissions",
    )

    t0 = time.time()
    last_text = ""

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)
                    last_text = block.text

    elapsed = round(time.time() - t0, 2)

    print(f"\n{'='*70}")
    print(f"Distill complete in {elapsed}s")
    print(f"{'='*70}")

    # Check outputs exist
    helper_path = os.path.join(output_dir, "helper.py")
    solutions_path = os.path.join(output_dir, "solutions.md")

    for path in [helper_path, solutions_path]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  OK: {path} ({size} bytes)")
        else:
            print(f"  WARNING: {path} not found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DABStep distill (Claude Agent SDK)")
    parser.add_argument(
        "--traces-dir",
        default=DEFAULT_TRACES_DIR,
        help="Directory with learning trace .json and .py files",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write helper.py and solutions.md",
    )
    parser.add_argument(
        "--data-dir",
        default=DATA_DIR,
        help="Directory with data files (payments.csv, fees.json, etc.)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: claude agent SDK default)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=60,
        help="Max agent turns",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
