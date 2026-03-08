#!/usr/bin/env python3
"""Post-process JSONL file to extract string part from 'string:number' answers."""

import argparse
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


def round_if_delta_question(question, answer):
    """Round numeric answer to 2 decimal places for fee delta questions."""
    if not question or not re.search(r'\bwhat delta\b', question, re.IGNORECASE):
        return answer
    try:
        val = float(answer)
        return round(val, 2)
    except (ValueError, TypeError):
        return answer


def load_tasks_questions(tasks_file='data/tasks.json'):
    """Load task_id -> question mapping from tasks.json."""
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)
    return {str(t['task_id']): t.get('question', '') for t in tasks}


def process_jsonl(input_file: str, output_file: str, extract_string=True):
    """Process JSONL file and write results to new file."""
    questions = load_tasks_questions()
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if not line.strip():
                continue
            data = json.loads(line)
            if 'agent_answer' in data:
                data['agent_answer'] = extract_nested_answer(data['agent_answer'])
                if extract_string:
                    data['agent_answer'] = extract_string_part(data['agent_answer'])
                question = data.get('question', '') or questions.get(str(data.get('task_id', '')), '')
                data['agent_answer'] = round_if_delta_question(
                    question, data['agent_answer'])
            f_out.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post-process JSONL answers.')
    parser.add_argument('input', help='Input JSONL file')
    parser.add_argument('output', help='Output JSONL file')
    parser.add_argument('--no-extract-string', action='store_true',
                        help='Disable extract_string_part processing')
    args = parser.parse_args()

    process_jsonl(args.input, args.output, extract_string=not args.no_extract_string)
    print(f"Processed {args.input} -> {args.output}")
