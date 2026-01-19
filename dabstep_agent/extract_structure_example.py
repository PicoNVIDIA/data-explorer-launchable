#!/usr/bin/env python3
"""
Example script: Extract structure from CSV and JSON files using DataScienceAgent.

This script demonstrates using the agent with only the Python execution tool
to extract file structures in a standardized JSON format.
"""

import os
import json
from dotenv import load_dotenv
from agent import DataScienceAgent
from tools import execute_python_code_tool

load_dotenv()

# Data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "context")


def extract_file_structure(agent: DataScienceAgent, file_path: str) -> dict:
    """
    Use the agent to extract structure from a CSV or JSON file.

    Args:
        agent: The DataScienceAgent instance
        file_path: Path to the file to analyze

    Returns:
        Dictionary containing the file structure
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == ".csv":
        prompt = f"""
Read the CSV file at '{file_path}' and output its structure as JSON with this exact format:
{{
  "file_type": "csv",
  "file_path": "{file_path}",
  "columns": ["col1", "col2", ...],
  "sample_row": {{"col1": "value1", "col2": "value2", ...}}
}}

Use pandas to read the file, get the column names, and the first row as a dict.
Print ONLY the JSON output, nothing else.
"""
    elif file_ext == ".json":
        prompt = f"""
Read the JSON file at '{file_path}' and output its structure as JSON with this exact format:
{{
  "file_type": "json",
  "file_path": "{file_path}",
  "structure_type": "array_of_objects" or "object" or "array_of_primitives",
  "keys": ["field1", "field2", ...],
  "sample_record": {{...first record or top-level object...}}
}}

Determine the structure_type based on what the JSON contains:
- "array_of_objects" if it's a list of dicts
- "object" if it's a single dict
- "array_of_primitives" if it's a list of simple values

Print ONLY the JSON output, nothing else.
"""
    else:
        return {"error": f"Unsupported file type: {file_ext}"}

    result = agent.process_prompt(prompt)

    # Try to parse the result as JSON
    try:
        # Find JSON in the response (in case there's extra text)
        result = result.strip()
        # Find the first { and last }
        start = result.find('{')
        end = result.rfind('}') + 1
        if start != -1 and end > start:
            json_str = result[start:end]
            return json.loads(json_str)
        return {"error": "No JSON found in response", "raw": result}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}", "raw": result}


def main():
    # Initialize agent with only the Python execution tool
    api_key = os.environ.get("NVIDIA_API_KEY")
    agent = DataScienceAgent(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        model="nvidia/nemotron-3-nano-30b-a3b",
        tools=[execute_python_code_tool],  # Only Python tool
        verbose=True,
        stream=True,
        skip_final_response=True,  # Return raw output directly
        max_iterations=5
    )

    # Find all CSV and JSON files in the data directory
    files_to_process = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(('.csv', '.json')):
            files_to_process.append(os.path.join(DATA_DIR, filename))

    print(f"Found {len(files_to_process)} files to process:")
    for f in files_to_process:
        print(f"  - {os.path.basename(f)}")
    print()

    # Extract structure from each file
    results = {}
    for file_path in files_to_process:
        print(f"\n{'='*70}")
        print(f"Processing: {os.path.basename(file_path)}")
        print('='*70)

        structure = extract_file_structure(agent, file_path)
        results[os.path.basename(file_path)] = structure

        # Reset conversation for next file (but keep execution environment)
        agent.reset_conversation()

    # Print final results
    print("\n" + "="*70)
    print("EXTRACTED STRUCTURES")
    print("="*70)
    print(json.dumps(results, indent=2, default=str))

    # Optionally save to file
    output_path = os.path.join(DATA_DIR, "file_structures.json")
    with open(output_path, 'w') as f:
        json.dump(results, indent=2, fp=f, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
