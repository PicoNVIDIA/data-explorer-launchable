#!/bin/bash

INPUT_FILE="data/tasks.json"
OUTPUT_FILE="results.jsonl"

# Extract task IDs from JSON
task_ids=$(jq -r '.[].task_id' "$INPUT_FILE")

for task_id in $task_ids; do
    echo "Running task: $task_id"
    python dabstep_agent/run.py --task-id "$task_id" --output "$OUTPUT_FILE" --input "$INPUT_FILE"
done

echo "All tasks completed. Results saved to $OUTPUT_FILE"
