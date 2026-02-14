#!/usr/bin/env python3
"""Post-process JSONL file to extract string part from 'string:number' answers."""

import json
import re
import sys


def extract_nested_answer(answer):
    """Extract the real answer if it's wrapped in a string containing JSON with agent_answer."""
    if not isinstance(answer, str):
        return answer

    # Look for embedded JSON with "agent_answer" key in the string
    match = re.search(r'\{[^{}]*"agent_answer"\s*:\s*(.+?)\s*\}', answer, re.DOTALL)
    if match:
        try:
            # Parse the entire matched JSON object
            json_str = match.group(0)
            parsed = json.loads(json_str)
            if 'agent_answer' in parsed:
                return parsed['agent_answer']
        except json.JSONDecodeError:
            pass

    return answer


def extract_string_part(answer: str) -> str:
    """If answer is 'string:number', return just the string part."""
    if not isinstance(answer, str):
        return answer

    # Match pattern: string followed by colon and number
    match = re.match(r'^([^:]+):(-?\d+\.?\d*)$', answer.strip())
    if match:
        return match.group(1)
    return answer


def process_jsonl(input_file: str, output_file: str):
    """Process JSONL file and write results to new file."""
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            if 'agent_answer' in data:
                data['agent_answer'] = extract_nested_answer(data['agent_answer'])
                data['agent_answer'] = extract_string_part(data['agent_answer'])
            f_out.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python postprocess.py <input.jsonl> <output.jsonl>")
        sys.exit(1)

    process_jsonl(sys.argv[1], sys.argv[2])
    print(f"Processed {sys.argv[1]} -> {sys.argv[2]}")
