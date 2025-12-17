import asyncio
import argparse
import csv
import os
import time
import tempfile
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from nat.runtime.loader import load_workflow

load_dotenv()

DABSTEP_PROMPT = """
IMPORTANT - Answer formatting:
  - When a question asks "What percentage..." or "What is the X rate...", return the percentage VALUE (e.g., 73.15), NOT the decimal proportion (e.g., 0.7315). Multiply by 100 if needed.
  - When a question asks for a "ratio", return the decimal value unless otherwise specified.
  - Always re-read the question and guidelines before providing your final answer to ensure the format matches.
"""

BASE_CONFIG_PATH = "src/data_explorer_agent/configs/config.yml"


def create_task_config(task_id: str, notebook_path: str) -> str:
    """Create a temporary config file with the task-specific notebook path."""
    with open(BASE_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Update notebook path for this task
    config["function_groups"]["notebook_function_group"]["notebook_path"] = notebook_path

    # Write to a temporary file
    temp_config = tempfile.NamedTemporaryFile(
        mode='w', suffix='.yml', delete=False, prefix=f'task_{task_id}_'
    )
    yaml.dump(config, temp_config)
    temp_config.close()
    return temp_config.name


async def run_task_async(config_path: str, user_query: str) -> str:
    """Run a single task using the NAT workflow."""
    async with load_workflow(config_path) as workflow:
        async with workflow.run(user_query) as runner:
            return await runner.result()


def process_task(task, dataset_path):
    """Process a single task and return the result."""
    task_id = task["task_id"]
    question = task["question"]
    guidelines = task["guidelines"]
    answer = task["answer"]
    level = task["level"]

    notebook_path = f"./app_notebooks/task_{task_id}.ipynb"
    start_time = time.time()
    temp_config_path = None

    try:
        # Create task-specific config
        temp_config_path = create_task_config(task_id, notebook_path)

        # Build the full query with dataset path info
        user_query = f"Dataset files available: {dataset_path}\n\n{question}\n\n{guidelines}\n\n{DABSTEP_PROMPT}"

        # Run the async workflow
        generated_answer = asyncio.run(run_task_async(temp_config_path, user_query))

        elapsed = time.time() - start_time
        return {
            "task_id": task_id,
            "question": question,
            "guidelines": guidelines,
            "answer": answer,
            "generated_answer": generated_answer,
            "level": level,
            "success": True,
            "elapsed_time": elapsed
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "task_id": task_id,
            "question": question,
            "guidelines": guidelines,
            "answer": answer,
            "generated_answer": f"ERROR: {e}",
            "level": level,
            "success": False,
            "elapsed_time": elapsed
        }
    finally:
        # Clean up temporary config file
        if temp_config_path and os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=str, help="Run only this task_id")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of tasks per batch")
    parser.add_argument("--workers", type=int, default=3, help="Number of parallel workers per batch")
    args = parser.parse_args()

    # Login using e.g. `huggingface-cli login` to access this dataset
    ds = load_dataset("adyen/DABstep", "tasks", split="default")  # split: dev or default
    dataset_path = "./data/acquirer_countries.csv, ./data/fees.json, ./data/manual.md, ./data/merchant_category_codes.csv, ./data/merchant_data.json, ./data/payments-readme.md, ./data/payments.csv"
    submission_file = f"submission_{ds.split}.csv"

    # Ensure app_notebooks directory exists
    os.makedirs("./app_notebooks", exist_ok=True)

    # Sort tasks: easy first, then hard
    level_order = {"easy": 0, "hard": 1}
    ds = sorted(ds, key=lambda x: level_order.get(x["level"], 2))

    # Read existing task_ids
    existing_task_ids = set()
    if os.path.exists(submission_file):
        with open(submission_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_task_ids.update({row["task_id"] for row in reader})

    # Filter tasks
    tasks_to_process = []
    for row in ds:
        task_id = row["task_id"]

        # Filter by task_id if specified
        if args.task_id and task_id != args.task_id:
            continue

        # Skip if already processed
        if task_id in existing_task_ids:
            print(f"Task {task_id} already in {submission_file}, skipping")
            continue

        tasks_to_process.append(row)

    print(f"Processing {len(tasks_to_process)} tasks with batch_size={args.batch_size}, workers={args.workers}")

    # Ensure submission file exists and has header
    write_header = not os.path.exists(submission_file)

    # Track overall progress
    total_tasks = len(tasks_to_process)
    completed_tasks = 0
    successful_tasks = 0
    failed_tasks = 0
    total_time = 0.0
    overall_start = time.time()

    # Create overall progress bar
    pbar = tqdm(total=total_tasks, desc="Overall Progress", unit="task")

    # Process tasks in batches
    for batch_start in range(0, len(tasks_to_process), args.batch_size):
        batch = tasks_to_process[batch_start:batch_start + args.batch_size]
        batch_num = batch_start // args.batch_size + 1
        total_batches = (len(tasks_to_process) + args.batch_size - 1) // args.batch_size

        pbar.set_description(f"Batch {batch_num}/{total_batches}")

        # Process batch in parallel
        results = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_task, task, dataset_path): task for task in batch}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                # Update progress
                completed_tasks += 1
                total_time += result["elapsed_time"]
                if result["success"]:
                    successful_tasks += 1
                else:
                    failed_tasks += 1

                # Update progress bar
                pbar.update(1)
                avg_time = total_time / completed_tasks
                pbar.set_postfix({
                    "success": successful_tasks,
                    "failed": failed_tasks,
                    "avg_time": f"{avg_time:.1f}s"
                })

        # Write batch results to CSV
        with open(submission_file, "a", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["task_id", "question", "guidelines", "answer", "generated_answer", "level"])
                write_header = False
            for result in results:
                writer.writerow([
                    result["task_id"],
                    result["question"],
                    result["guidelines"],
                    result["answer"],
                    result["generated_answer"],
                    result["level"]
                ])
            f.flush()

    pbar.close()

    # Print final summary
    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*50}")
    print(f"COMPLETED: {completed_tasks}/{total_tasks} tasks")
    if completed_tasks > 0:
        print(f"  Success: {successful_tasks} | Failed: {failed_tasks}")
        print(f"  Total time: {overall_elapsed:.1f}s | Avg per task: {total_time/completed_tasks:.1f}s")
        print(f"  Results saved to: {submission_file}")
    else:
        print(f"  No tasks to process")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
