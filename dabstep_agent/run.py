import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from qa_agent import QAAgent, load_question
from extract_structure_example import (
    create_structure_extraction_agent,
    extract_all_structures,
    format_structures_for_prompt
)
from dotenv import load_dotenv

load_dotenv()

# Global variable to hold pre-extracted file structures
_FILE_STRUCTURES = None


def extract_file_structures_once(data_dir: str = "data/context") -> str:
    """Extract file structures once and cache them."""
    global _FILE_STRUCTURES
    if _FILE_STRUCTURES is None:
        print("Pre-extracting file structures...")
        structure_agent = create_structure_extraction_agent()
        structures = extract_all_structures(structure_agent, data_dir)
        _FILE_STRUCTURES = format_structures_for_prompt(structures)
        print("File structures extracted and cached.\n")
    return _FILE_STRUCTURES


def solve_single_task(task_id: int, file_structures: str = None) -> dict:
    """Solve a single task using QAAgent and return the result."""
    try:
        # Create QAAgent instance
        agent = QAAgent(
            data_dir="data/context",
            tasks_file="data/tasks_dev.json",
            file_structures=file_structures,
            default_search_terms=['null'],
            verbose=False
        )

        question = agent.get_question(task_id)
        answer = agent.solve(task_id)

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
    # Pre-extract file structures once before parallel execution
    file_structures = extract_file_structures_once()

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
        # Submit all tasks with pre-extracted structures
        future_to_task = {
            executor.submit(solve_single_task, task_id, file_structures): task_id
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
    """Solve a single task using QAAgent."""
    # Pre-extract file structures
    file_structures = extract_file_structures_once()

    # Create and use QAAgent
    agent = QAAgent(
        data_dir="data/context",
        tasks_file="data/tasks_dev.json",
        file_structures=file_structures,
        default_search_terms=['null'],
        verbose=False
    )

    return agent.solve(task_id)


def learn_single(task_id: int, gt_answer: str):
    """Learn from a single task using QAAgent with ground truth answer."""
    # Pre-extract file structures
    file_structures = extract_file_structures_once()

    # Create and use QAAgent
    agent = QAAgent(
        data_dir="data/context",
        tasks_file="data/tasks_dev.json",
        file_structures=file_structures,
        default_search_terms=['null'],
        verbose=False
    )

    answer, rule = agent.learn(task_id, gt_answer)

    # Print extracted rule separately for easy access
    if rule:
        print("\n" + "=" * 70)
        print("EXTRACTED RULE/INSIGHT:")
        print("=" * 70)
        print(rule)
        print("=" * 70 + "\n")

    return answer, rule


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DABStep Task Solver")
    parser.add_argument("--parallel", action="store_true", help="Run all tasks in parallel")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers (default: 4)")
    parser.add_argument("--task", type=int, default=None, help="Solve a single task by ID")
    parser.add_argument("--output", type=str, default=None, help="Output file for results")
    parser.add_argument("--learn", action="store_true", help="Learn mode: find code that produces ground truth answer")
    parser.add_argument("--gt-answer", type=str, default=None, help="Ground truth answer for learn mode")

    args = parser.parse_args()

    if args.parallel:
        solve_all_tasks_parallel(max_workers=args.workers, output_file=args.output)
    elif args.learn:
        if args.task is None:
            print("Error: --learn requires --task to specify the task ID")
            exit(1)
        if args.gt_answer is None:
            # Try to get gt_answer from the task's answer field
            with open("data/tasks_dev.json", 'r') as f:
                questions = json.load(f)
            if args.task < len(questions) and "answer" in questions[args.task]:
                gt_answer = questions[args.task]["answer"]
                print(f"Using answer from task file: {gt_answer}")
            else:
                print("Error: --learn requires --gt-answer or task must have 'answer' field")
                exit(1)
        else:
            gt_answer = args.gt_answer
        learn_single(args.task, gt_answer)
    elif args.task is not None:
        solve_single(args.task)
    else:
        # Default: solve task 1
        solve_single(1)
