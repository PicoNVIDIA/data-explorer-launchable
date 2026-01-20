#!/usr/bin/env python3
"""
Simple test to verify the DataScienceAgent can analyze JSON data.
Tests the agent's ability to write and execute Python code to query fees.json.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dabstep_agent.agent import DataScienceAgent

load_dotenv()

# Default path to test data
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "context", "fees.json"
)


def create_test_agent():
    """Create a DataScienceAgent for testing."""
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


def run_query(query: str, data_path: str = DATA_PATH):
    """Run a custom query against the JSON data."""
    print("=" * 70)
    print("QUERY:")
    print(query)
    print("=" * 70)

    agent = create_test_agent()

    prompt = f"""
You have access to a JSON file at: {data_path}

Question: {query}

Write Python code to load the JSON file and answer the question.
"""

    print(f"\nFull prompt sent to agent:\n{prompt}\n")

    response = agent.process_prompt(prompt)

    print("\n" + "=" * 70)
    print("AGENT RESPONSE:")
    print("=" * 70)
    print(response)

    return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test DataScienceAgent JSON capabilities")
    parser.add_argument("--query", "-q", type=str,
                        help="Custom query to run against the JSON data")
    parser.add_argument("--data", "-d", type=str, default=DATA_PATH,
                        help=f"Path to JSON data file (default: {DATA_PATH})")
    # "which card_scheme has account_type H?"
    args = parser.parse_args()

    if args.query:
        run_query(args.query, args.data)
    else:
        # Default example query
        run_query("How many records are in the file?", args.data)
