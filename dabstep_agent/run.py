import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from agent import DataScienceAgent
from utils import solver, load_question
from dotenv import load_dotenv

load_dotenv()


def create_agent():
    """Create a new DataScienceAgent instance."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    agent = DataScienceAgent(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        max_iterations=100,
        model="nvidia/nemotron-3-nano-30b-a3b",
        verbose=True,
        stream=True,
        skip_final_response=False
    )
    agent.reset_conversation()
    return agent


def solve_single_task(task_id: int) -> dict:
    """Solve a single task and return the result."""
    try:
        agent = create_agent()
        dev_jsonl = "data/tasks_dev.json"
        question = load_question(dev_jsonl, index=task_id)

        answer = solver(agent, task_id)

        return {
            "task_id": question.get("task_id", task_id),
            "question": question["question"],
            "expected_answer": question.get("answer", "N/A"),
            "agent_answer": answer,
            "status": "success"
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "question": "N/A",
            "expected_answer": "N/A",
            "agent_answer": None,
            "status": "error",
            "error": str(e)
        }


def solve_all_tasks_parallel(max_workers: int = 4, output_file: str = None):
    """
    Solve all dabstep tasks in parallel and write results to a file.

    Args:
        max_workers: Number of parallel workers
        output_file: Output file path (default: results_<timestamp>.json)
    """
    # Load all questions to get the count
    dev_jsonl = "data/tasks_dev.json"
    with open(dev_jsonl, 'r') as f:
        questions = json.load(f)

    num_tasks = len(questions)
    print(f"Found {num_tasks} tasks to solve")

    # Generate output filename if not provided
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"results_{timestamp}.json"

    results = []
    completed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(solve_single_task, task_id): task_id
            for task_id in range(num_tasks)
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            task_id = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(f"\n[Progress: {completed}/{num_tasks}] Task {task_id} completed - Status: {result['status']}")
            except Exception as e:
                results.append({
                    "task_id": task_id,
                    "status": "error",
                    "error": str(e)
                })
                completed += 1
                print(f"\n[Progress: {completed}/{num_tasks}] Task {task_id} failed with exception: {e}")

    # Sort results by task_id for consistent ordering
    results.sort(key=lambda x: x.get("task_id", 0) if isinstance(x.get("task_id"), int) else 0)

    # Calculate summary statistics
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "error")

    output_data = {
        "summary": {
            "total_tasks": num_tasks,
            "successful": successful,
            "failed": failed,
            "timestamp": datetime.now().isoformat()
        },
        "results": results
    }

    # Write results to file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"All tasks completed!")
    print(f"Total: {num_tasks}, Successful: {successful}, Failed: {failed}")
    print(f"Results written to: {output_file}")
    print(f"{'=' * 70}")

    return output_data


def solve_single(task_id: int):
    """Solve a single task (original behavior)."""
    agent = create_agent()
    solver(agent, task_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DABStep Task Solver")
    parser.add_argument("--parallel", action="store_true", help="Run all tasks in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--task", type=int, default=None, help="Solve a single task by ID")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")

    args = parser.parse_args()

    if args.parallel:
        solve_all_tasks_parallel(max_workers=args.workers, output_file=args.output)
    elif args.task is not None:
        solve_single(args.task)
    else:
        # Default: solve task 1
        solve_single(1)
