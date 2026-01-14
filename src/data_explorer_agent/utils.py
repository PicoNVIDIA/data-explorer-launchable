import json


def print_token_usage(traces_file: str):
    """Print token usage from each LLM call in the traces file."""
    totals = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
    with open(traces_file) as f:
        for line in f:
            p = json.loads(line).get("payload", {})
            if p.get("event_type") == "LLM_END":
                t = p.get("usage_info", {}).get("token_usage", {})
                totals["calls"] += 1
                totals["prompt"] += t.get("prompt_tokens", 0)
                totals["completion"] += t.get("completion_tokens", 0)
                totals["total"] += t.get("total_tokens", 0)
                print(f"  Call {totals['calls']}: prompt={t.get('prompt_tokens', 0)}, completion={t.get('completion_tokens', 0)}")
    print(f"\nTotal: {totals['calls']} calls, {totals['prompt']} prompt, {totals['completion']} completion, {totals['total']} total tokens")
