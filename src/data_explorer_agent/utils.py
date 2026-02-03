import json


def print_token_usage(traces_file: str):
    """Print token usage from each LLM call in the traces file."""
    totals = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
    errors = 0
    with open(traces_file) as f:
        for line_num, line in enumerate(f, 1):
            try:
                p = json.loads(line).get("payload", {})
            except json.JSONDecodeError as e:
                errors += 1
                print(f"  Warning: skipping line {line_num} (invalid JSON: {e.msg})")
                continue
            if p.get("event_type") == "LLM_END":
                t = p.get("usage_info", {}).get("token_usage", {})
                totals["calls"] += 1
                totals["prompt"] += t.get("prompt_tokens", 0)
                totals["completion"] += t.get("completion_tokens", 0)
                totals["total"] += t.get("total_tokens", 0)
                print(f"  Call {totals['calls']}: prompt={t.get('prompt_tokens', 0)}, completion={t.get('completion_tokens', 0)}")
    if errors:
        print(f"\nSkipped {errors} invalid line(s)")
    print(f"Total: {totals['calls']} calls, {totals['prompt']} prompt, {totals['completion']} completion, {totals['total']} total tokens")
