"""
Test the python_executor tool with a NAT tool calling agent.

Usage:
    uv run python python_executor_example/run.py
    uv run python python_executor_example/run.py --query "What is 2+2?"
    uv run python python_executor_example/run.py --query "Load data/context/payments.csv and show the first 5 rows"
"""

import argparse
import asyncio

from nat.runtime.loader import load_workflow
from dotenv import load_dotenv

load_dotenv()

CONFIG = "python_executor_example/config.yml"

DEFAULT_QUERY = (
    "Do these steps ONE AT A TIME, each in a separate execute_python_code call:\n"
    "1. Create a dataframe with columns 'name' and 'score' for 5 students. Print it.\n"
    "2. Add a new column 'grade' based on score (A>=90, B>=80, C>=70, else F). Print the updated df.\n"
    "3. Using the same dataframe, compute the mean score per grade and print the result.\n"
    "Do NOT combine steps into one code block."
)


async def run_workflow(query: str) -> str:
    async with load_workflow(CONFIG) as workflow:
        async with workflow.run(query) as runner:
            return await runner.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test python_executor NAT tool")
    parser.add_argument("--query", type=str, default=DEFAULT_QUERY)
    args = parser.parse_args()

    print(f"Query: {args.query}\n{'=' * 70}\n")
    result = asyncio.run(run_workflow(args.query))
    print(f"\n{'=' * 70}\nFinal answer: {result}")
