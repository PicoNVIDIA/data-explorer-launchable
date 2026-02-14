#!/bin/bash

usage() {
    echo "Usage: $0 [INPUT_FILE] [NUM_PASSES]"
    echo ""
    echo "Run all tasks from a JSON file with multiple passes."
    echo ""
    echo "Arguments:"
    echo "  INPUT_FILE   Path to tasks JSON file (default: data/tasks.json)"
    echo "  NUM_PASSES   Number of passes to run (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0                          # Run data/tasks.json with 1 pass"
    echo "  $0 data/tasks_dev.json      # Run dev tasks with 1 pass"
    echo "  $0 data/tasks.json 3        # Run tasks with 3 passes"
    exit 0
}

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    usage
fi

INPUT_FILE="${1:-data/tasks.json}"
NUM_PASSES=${2:-1}

# Derive suffix from input filename (e.g., tasks_dev.json -> _dev, tasks.json -> "")
BASENAME=$(basename "$INPUT_FILE" .json)
SUFFIX="${BASENAME#tasks}"

# Extract task IDs from JSON
task_ids=$(jq -r '.[].task_id' "$INPUT_FILE")

for pass in $(seq 1 $NUM_PASSES); do
    echo "=========================================="
    echo "Starting pass $pass of $NUM_PASSES"
    echo "=========================================="

    WORKSPACE_DIR="workspace/pass${pass}${SUFFIX}"
    OUTPUT_FILE="results_pass${pass}${SUFFIX}.jsonl"

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
