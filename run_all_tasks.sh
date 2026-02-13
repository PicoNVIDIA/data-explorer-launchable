#!/bin/bash

INPUT_FILE="data/tasks.json"
NUM_PASSES=${1:-1}

# Extract task IDs from JSON
task_ids=$(jq -r '.[].task_id' "$INPUT_FILE")

for pass in $(seq 1 $NUM_PASSES); do
    echo "=========================================="
    echo "Starting pass $pass of $NUM_PASSES"
    echo "=========================================="

    WORKSPACE_DIR="workspace/pass${pass}"
    OUTPUT_FILE="results_pass${pass}.jsonl"

    # Clear output file for this pass
    > "$OUTPUT_FILE"

    for task_id in $task_ids; do
        echo "Running task: $task_id (pass $pass)"
        python dabstep_agent/run.py --task-id "$task_id" --output "$OUTPUT_FILE" --input "$INPUT_FILE" --workspace "$WORKSPACE_DIR"
    done

    echo "Pass $pass completed. Results saved to $OUTPUT_FILE, code saved to '$WORKSPACE_DIR'"
done

echo ""
echo "All $NUM_PASSES passes completed."
