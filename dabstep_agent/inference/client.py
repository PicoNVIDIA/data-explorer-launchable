"""
Client for DABStep Agent API.

Usage:
    uv run python dabstep_agent/client.py --input data/tasks.json --task-id 1712
    uv run python dabstep_agent/client.py --input data/tasks.json --task-id 1712,1810
    uv run python dabstep_agent/client.py --input data/tasks.json  # all tasks
    uv run python dabstep_agent/client.py --input data/tasks.json --task-id 1712 --url http://localhost:8080
"""

import argparse
import json
import sys
import time

import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="DABStep Agent API client")
    parser.add_argument("--input", required=True, help="Tasks JSON file")
    parser.add_argument("--task-id", default=None, help="Comma-separated task IDs (optional, default: all)")
    parser.add_argument("--url", default="http://localhost:8081", help="API base URL")
    parser.add_argument("--output", default=None, help="Output JSONL file (default: results_api.jsonl)")
    parser.add_argument("--no-timing", dest="timing", action="store_false", help="Disable per-task timing")
    parser.set_defaults(timing=True)
    args = parser.parse_args()

    with open(args.input) as f:
        tasks = json.load(f)

    if args.task_id:
        filter_ids = set(args.task_id.split(","))
        tasks = [t for t in tasks if t.get("task_id") in filter_ids]

    if not tasks:
        print("No matching tasks found.")
        sys.exit(1)

    output_file = args.output or "results_api.jsonl"
    endpoint = f"{args.url.rstrip('/')}/solve"

    print(f"Sending {len(tasks)} task(s) to {endpoint}")
    print(f"Output: {output_file}\n")

    for task in tqdm(tasks, desc="Solving tasks"):
        task_id = task.get("task_id", "?")
        tqdm.write(f"Task {task_id}: {task['question'][:100]}...")

        if args.timing:
            t0 = time.time()
        resp = requests.post(endpoint, json={
            "question": task["question"],
            "guidelines": task.get("guidelines", "N/A"),
        }, verify=False)
        resp.raise_for_status()
        result = resp.json()

        tqdm.write(f"  Answer: {result['agent_answer']}")
        tqdm.write(f"  Trace length: {len(result['reasoning_trace'])} messages")

        record = {
            "task_id": task_id,
            "agent_answer": result["agent_answer"],
            "reasoning_trace": result["reasoning_trace"],
        }
        if args.timing:
            elapsed = round(time.time() - t0, 2)
            tqdm.write(f"  Time: {elapsed}s")
            record["time_seconds"] = elapsed
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    print(f"Done. Results saved to {output_file}")


if __name__ == "__main__":
    main()
