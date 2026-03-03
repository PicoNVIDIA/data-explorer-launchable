"""
DABStep distill phase: synthesize learning traces into helper.py + solutions.md.

Reads the *.json (traces with questions/answers) and *.py (generated code)
from the learning traces directory, then prompts an LLM agent to:
  1. Identify common patterns, reusable functions, and key insights
  2. Write a clean helper.py module with shared utility functions
  3. Write a solutions.md with documented solution patterns and examples
  4. Verify every helper function against ground truth answers

Usage:
    uv run python dabstep_agent/distill.py
    uv run python dabstep_agent/distill.py --traces-dir dabstep_agent/workspace/learning_traces
    uv run python dabstep_agent/distill.py --output-dir dabstep_agent
"""

import argparse
import asyncio
import glob
import json
import os
import re
import time

import yaml
from nat.runtime.loader import load_workflow
from nat.plugins.langchain.agent.tool_calling_agent.agent import ToolCallAgentGraph
from dotenv import load_dotenv

load_dotenv()

DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(DIR, "distill_config.yml")
DATA_DIR = "data/context"


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


def load_file_structures(data_dir: str = DATA_DIR) -> str:
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
                lines.append(f"- {filename} (CSV):")
                lines.append(f"    Columns: {', '.join(structure.get('columns', []))}")
                lines.append(f"    Sample: {json.dumps(sample, default=str)}")
            elif file_type == "json":
                keys = structure.get("keys", [])
                sample = structure.get("sample_record", {})
                lines.append(f"- {filename} (JSON, {structure.get('structure_type', '')}):")
                lines.append(f"    Keys: {', '.join(keys)}")
                lines.append(f"    Sample: {json.dumps(sample, default=str)}")
        return "\n".join(lines)
    return ""


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

def build_distill_prompt(traces: list[dict], file_structures: str, output_dir: str) -> str:
    traces_text = "\n---\n".join(format_trace_summary(t) for t in traces)

    return f"""You are distilling messy learning traces into clean, reusable code artifacts.

## Context

Below are learning traces from an agent that solved DABStep fee-calculation questions.
Each trace has a question, ground truth answer, the agent's answer, and the Python code the agent wrote.
Some traces are CORRECT, some FAILED. Learn from both — understand what worked and what went wrong.

## Data Files (in '{DATA_DIR}/')

{file_structures}

There is also a manual at '{DATA_DIR}/manual.md' with term definitions.

## Learning Traces

{traces_text}

## Your Task

Analyze ALL the traces above and produce TWO files:

### File 1: `{output_dir}/helper.py`

A Python module with clean, reusable functions. It must include:

1. **Data loading functions**: load_fees(), load_payments(), load_merchants(), load_acquirer_countries(), get_merchant_info(name)
2. **Field matching functions**: Functions that implement the matching semantics from fees.json:
   - `matches_list_field(field_value, target)` — null or empty list = applies to ALL values
   - `matches_bool_field(field_value, target)` — null = applies to all
   - `matches_capture_delay(fee_delay, merchant_delay)` — handles '<3', '3-5', '>5', 'immediate', 'manual'
   - `matches_monthly_volume(rule_vol, monthly_vol)` — handles '<100k', '100k-1m', '>10m' etc.
   - `matches_fraud_level(rule_fraud, fraud_pct)` — handles '<7.2%', '7.7%-8.3%', '>8.3%' etc.
3. **Range parsing**: parse_volume_value(), parse_volume_range(), parse_fraud_value(), parse_fraud_range()
4. **Fee calculation**: calculate_fee(fixed_amount, rate, transaction_value) — formula: fixed_amount + rate * transaction_value / 10000
5. **Transaction filtering**: filter_merchant_transactions(df, merchant, year, month), get_month_day_range(month)
6. **Monthly metrics**: calculate_monthly_metrics(df) — returns volume, fraud_volume, fraud_rate_pct
7. **Intracountry flag**: add_intracountry_flag(df) — uses per-transaction acquirer_country column from payments.csv
8. **Composite matching**: matches_fee_rule() and find_matching_fees() that combine all the above

Key rules to encode (learned from the traces):
- null or [] in a list field means "applies to ALL" — not "no match"
- capture_delay from merchant_data.json should be passed as-is (string), not converted
- intracountry is determined per-transaction by comparing issuing_country to acquirer_country
- Monthly volume and fraud level must be computed per-month, not annually
- Fee formula: fixed_amount + rate * transaction_value / 10000

### File 2: `{output_dir}/solutions.md`

A markdown document with:

1. **Key Insights** at the top — bullet points of critical rules and gotchas learned from the traces
   (e.g., null/empty = wildcard, monthly metrics per-month not annual, intracountry per-transaction, etc.)
2. **Helper Module reference** — list all function signatures from helper.py
3. **Example Solutions** — for each solved question, document:
   - The question
   - Data sources used
   - Approach (1-3 sentences)
   - Clean, minimal code using helper.py functions (not the messy trace code)
   - For FAILED traces, write the code that WOULD produce the correct answer

## Instructions

1. First, use run_bash to read the manual.md and understand the domain (grep for key terms: fee, fraud, ACI, intracountry, capture_delay)
2. Use run_bash to read the actual data files (sample a few records from fees.json, payments.csv)
3. Write helper.py using execute_python_code — write it to '{output_dir}/helper.py'
4. **TEST every function** against the ground truths from the traces:
   - Task 5: highest issuing country = NL
   - Task 49: top fraud ip_country = BE (by fraud rate = fraud_vol/total_vol)
   - Task 1273: avg GlobalCard credit fee for 10 EUR = 0.120132 (include is_credit=True AND is_credit=null rules)
   - Task 1305: avg GlobalCard fee for account_type H, MCC 5812, 10 EUR = 0.123217
   - Task 1464: fee IDs for account_type=R, aci=B = (long sorted list starting with 1,2,5,6,8...)
   - Task 1681: fee IDs for Belles_cookbook_store on Jan 10, 2023 = 286,381,454,473,477,536,572,709,741,813
   - Task 1753: fee IDs for Belles in March 2023 = 34 IDs
   - Task 1871: delta if fee 384 rate changed to 1 for Belles in Jan = -0.94000000000005
5. Fix any function that produces wrong results. Iterate until all ground truths match.
6. Write solutions.md to '{output_dir}/solutions.md'

Write the files using execute_python_code with open()/write() calls.
Do NOT proceed to the next step until the current step's tests pass.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main(args):
    traces_dir = args.traces_dir
    output_dir = args.output_dir

    print(f"Loading traces from: {traces_dir}")
    traces = load_traces(traces_dir)
    print(f"Found {len(traces)} traces: {[t['task_id'] for t in traces]}")
    print(f"  Correct: {sum(1 for t in traces if t['match'])}")
    print(f"  Failed:  {sum(1 for t in traces if not t['match'])}")

    file_structures = load_file_structures()
    prompt = build_distill_prompt(traces, file_structures, output_dir)

    print(f"\nOutput dir: {output_dir}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"\n{'='*70}")
    print("Starting distill agent...")
    print(f"{'='*70}\n")

    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    async with load_workflow(CONFIG) as workflow:
        async with workflow.run(prompt) as runner:
            result = await runner.result()

    elapsed = round(time.time() - t0, 2)

    print(f"\n{'='*70}")
    print(f"Distill complete in {elapsed}s")
    print(f"{'='*70}")
    print(f"Agent output: {result}")

    # Check outputs exist
    helper_path = os.path.join(output_dir, "helper.py")
    solutions_path = os.path.join(output_dir, "solutions.md")

    for path in [helper_path, solutions_path]:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {path}: {size} bytes")
        else:
            print(f"  WARNING: {path} not found!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DABStep distill: traces -> helper.py + solutions.md")
    parser.add_argument(
        "--traces-dir",
        default=os.path.join(DIR, "workspace", "learning_traces"),
        help="Directory with learning trace .json and .py files",
    )
    parser.add_argument(
        "--output-dir",
        default=DIR,
        help="Directory to write helper.py and solutions.md",
    )
    args = parser.parse_args()

    asyncio.run(main(args))
