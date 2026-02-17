#!/usr/bin/env python3
"""
Ensemble multiple results*.jsonl files using majority voting.
"""

import json
import glob
from collections import Counter


def load_results(filepath):
    """Load a jsonl file and return a dict mapping task_id to agent_answer."""
    results = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                results[data['task_id']] = data['agent_answer']
    return results


def ensemble_majority_vote(all_results):
    """
    Ensemble results using majority voting.

    Args:
        all_results: List of dicts, each mapping task_id to agent_answer

    Returns:
        Dict mapping task_id to the ensembled agent_answer
    """
    all_task_ids = set()
    for results in all_results:
        all_task_ids.update(results.keys())

    ensembled = {}

    for task_id in all_task_ids:
        answers = []
        for results in all_results:
            if task_id in results:
                answers.append(results[task_id])

        if not answers:
            continue

        # Convert lists to tuples so they are hashable for Counter
        hashable_answers = [tuple(a) if isinstance(a, list) else a for a in answers]
        counter = Counter(hashable_answers)
        winner = counter.most_common(1)[0][0]
        # Convert tuples back to lists
        ensembled[task_id] = list(winner) if isinstance(winner, tuple) else winner

    return ensembled


def main():
    results_files = sorted(f for f in glob.glob('results*.jsonl') if 'ensembled' not in f)
    print(f"Found {len(results_files)} results files: {results_files}")

    all_results = []
    for filepath in results_files:
        results = load_results(filepath)
        all_results.append(results)
        print(f"Loaded {filepath}: {len(results)} tasks")

    ensembled = ensemble_majority_vote(all_results)
    print(f"\nEnsembled results: {len(ensembled)} tasks")

    output_path = 'results_ensembled.jsonl'
    with open(output_path, 'w') as f:
        for task_id, agent_answer in ensembled.items():
            f.write(json.dumps({'task_id': task_id, 'agent_answer': agent_answer}) + '\n')

    print(f"Written to {output_path}")


if __name__ == '__main__':
    main()
