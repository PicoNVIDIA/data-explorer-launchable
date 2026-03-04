#!/usr/bin/env python3
"""Score predictions against dev set ground truth."""

import json
import sys
from metric import question_scorer


def load_jsonl(filepath):
    """Load JSONL file."""
    with open(filepath, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 5:
        print("Usage: python score_dev.py <predictions.jsonl> [ground_truth.jsonl] [--first N]")
        sys.exit(1)

    # Parse --first N argument
    first_n = None
    args = sys.argv[1:]
    filtered_args = []
    i = 0
    while i < len(args):
        if args[i] == '--first' and i + 1 < len(args):
            first_n = int(args[i + 1])
            i += 2
        else:
            filtered_args.append(args[i])
            i += 1

    # Load ground truth and predictions
    predictions = load_jsonl(filtered_args[0])
    use_custom_gt = len(filtered_args) == 2
    ground_truth_file = filtered_args[1] if use_custom_gt else 'data/dev.jsonl'
    ground_truth = load_jsonl(ground_truth_file)

    if first_n is not None:
        predictions = predictions[:first_n]

    # Load difficulty levels from all.jsonl
    all_tasks = load_jsonl('data/all.jsonl')
    difficulty_dict = {int(item['task_id']): item.get('level', 'unknown') for item in all_tasks}

    # Create dictionaries by task_id
    gt_dict = {int(item['task_id']): item for item in ground_truth}
    question_dict = {int(item['task_id']): item.get('question', '') for item in all_tasks}
    pred_dict = {}
    for item in predictions:
        tid = int(item['task_id'])
        ans = item.get('answer', item.get('agent_answer', item.get('prediction', '')))
        pred_dict[tid] = ans

    # Use 'agent_answer' key for custom ground truth files, 'answer' for default
    answer_key = 'agent_answer' if use_custom_gt else 'answer'

    # Identify tasks requiring string:number format (for lowest score calculation)
    tasks_meta = json.load(open('data/tasks.json'))
    format_task_ids = set()
    for t in tasks_meta:
        if t.get('guidelines') and '}:{' in t['guidelines']:
            format_task_ids.add(int(t['task_id']))

    # Find ground truth format failures: tasks requiring string:number but GT only has the string part
    gt_format_failures = set()
    for task_id, gt_item in gt_dict.items():
        if task_id not in format_task_ids:
            continue
        gt_ans = str(gt_item.get(answer_key, '')).strip()
        parts = gt_ans.split(':')
        ok = len(parts) == 2 and len(parts[0].strip()) > 0
        if ok:
            try:
                float(parts[1])
            except ValueError:
                ok = False
        if not ok:
            gt_format_failures.add(task_id)

    # Score each task
    correct = 0
    correct_low = 0  # lowest possible score
    total = 0
    errors = []
    skipped = 0

    # Track accuracy by difficulty
    difficulty_stats = {
        'easy': {'correct': 0, 'total': 0},
        'hard': {'correct': 0, 'total': 0}
    }
    difficulty_stats_low = {
        'easy': {'correct': 0, 'total': 0},
        'hard': {'correct': 0, 'total': 0}
    }

    for task_id, gt_item in gt_dict.items():
        # Skip tasks without predictions
        if task_id not in pred_dict:
            skipped += 1
            continue

        gt_answer = gt_item[answer_key]
        pred_answer = pred_dict[task_id]
        difficulty = difficulty_dict.get(task_id, gt_item.get('level', 'unknown'))
        total += 1
        is_correct = question_scorer(str(pred_answer), str(gt_answer))

        # For lowest score: if GT has format failure, treat as wrong
        is_correct_low = is_correct and (task_id not in gt_format_failures)

        # Update difficulty-specific stats
        if difficulty in difficulty_stats:
            difficulty_stats[difficulty]['total'] += 1
            difficulty_stats_low[difficulty]['total'] += 1
            if is_correct:
                difficulty_stats[difficulty]['correct'] += 1
            if is_correct_low:
                difficulty_stats_low[difficulty]['correct'] += 1

        if is_correct:
            correct += 1
        if is_correct_low:
            correct_low += 1

        if not is_correct:
            errors.append({
                'task_id': task_id,
                'predicted': pred_answer,
                'ground_truth': gt_answer,
                'difficulty': difficulty
            })

    # Print results
    accuracy = correct / total if total > 0 else 0
    accuracy_low = correct_low / total if total > 0 else 0
    print(f"Highest possible: {correct}/{total} = {accuracy:.2%}")
    print(f"Lowest  possible: {correct_low}/{total} = {accuracy_low:.2%}")
    if gt_format_failures:
        matched_failures = gt_format_failures & set(pred_dict.keys())
        print(f"  (GT format failures in scored tasks: {len(matched_failures)})")

    # Print accuracy by difficulty
    for difficulty in ['easy', 'hard']:
        stats = difficulty_stats[difficulty]
        stats_low = difficulty_stats_low[difficulty]
        if stats['total'] > 0:
            diff_accuracy = stats['correct'] / stats['total']
            diff_accuracy_low = stats_low['correct'] / stats_low['total']
            print(f"  {difficulty.capitalize()}: high={stats['correct']}/{stats['total']}={diff_accuracy:.2%}  low={stats_low['correct']}/{stats_low['total']}={diff_accuracy_low:.2%}")

    if skipped > 0:
        print(f"Skipped {skipped} task(s) with no predictions")

    # Print error details
    if errors and not use_custom_gt:
        print(f"\n{len(errors)} incorrect predictions:")
        print("-" * 80)
        for error in errors:
            print(f"\nTask ID: {error['task_id']} (Difficulty: {error['difficulty']})")
            print(f"  Predicted:    {error['predicted']}")
            print(f"  Ground Truth: {error['ground_truth']}")

    # Save incorrect task IDs to JSON file
    if errors:
        wrong_task_ids = [error['task_id'] for error in errors]
        with open('wrong_task_ids.json', 'w') as f:
            json.dump(wrong_task_ids, f, indent=2)
        print(f"\nSaved {len(wrong_task_ids)} incorrect task IDs to wrong_task_ids.json")

    # Save detailed wrong tasks with question and both answers
    if errors:
        wrong_tasks = []
        for error in errors:
            wrong_tasks.append({
                'task_id': error['task_id'],
                'question': question_dict.get(error['task_id'], ''),
                'predicted': error['predicted'],
                'ground_truth': error['ground_truth'],
                'difficulty': error['difficulty']
            })
        with open('wrong_tasks_detailed.json', 'w') as f:
            json.dump(wrong_tasks, f, indent=2)
        print(f"Saved {len(wrong_tasks)} wrong tasks with details to wrong_tasks_detailed.json")


if __name__ == '__main__':
    main()
