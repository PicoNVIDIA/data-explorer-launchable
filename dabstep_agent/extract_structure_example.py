#!/usr/bin/env python3
"""
Extract structure from CSV and JSON files using DataScienceAgent.

This module provides functions to extract file structures in a standardized JSON format.
Can be used as a library or run as a standalone script.
"""

import os
import json
from typing import Dict, Optional
from dotenv import load_dotenv
from agent import DataScienceAgent
from tools import execute_python_code_tool

load_dotenv()


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


def extract_all_structures(agent: DataScienceAgent, data_dir: str) -> Dict[str, dict]:
    """
    Extract structures from all CSV and JSON files in a directory.

    Args:
        agent: The DataScienceAgent instance
        data_dir: Path to the directory containing data files

    Returns:
        Dictionary mapping filenames to their structure dictionaries
    """
    output_path = os.path.join(data_dir, "file_structures.json")

    # If cached file exists, load and return it
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            return json.load(f)

    results = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(('.csv', '.json')):
            file_path = os.path.join(data_dir, filename)
            structure = extract_file_structure(agent, file_path)
            results[filename] = structure
            agent.reset_conversation()

    # Save results to cache file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def format_structures_for_prompt(structures: Dict[str, dict]) -> str:
    """
    Format extracted structures as a string for injection into prompts.

    Args:
        structures: Dictionary mapping filenames to structure dictionaries

    Returns:
        Formatted string describing all file structures
    """
    lines = []

    for filename, structure in structures.items():
        if "error" in structure:
            lines.append(f"- {filename}: (error extracting structure)")
            continue

        file_type = structure.get("file_type", "unknown")

        if file_type == "csv":
            columns = structure.get("columns", [])
            sample = structure.get("sample_row", {})
            lines.append(f"- {filename} (CSV):")
            lines.append(f"    Columns: {', '.join(columns)}")
            lines.append(f"    Sample row: {json.dumps(sample, default=str)}")
        elif file_type == "json":
            keys = structure.get("keys", [])
            structure_type = structure.get("structure_type", "unknown")
            sample = structure.get("sample_record", {})
            lines.append(f"- {filename} (JSON, {structure_type}):")
            lines.append(f"    Keys: {', '.join(keys)}")
            lines.append(f"    Sample record: {json.dumps(sample, default=str)}")

    return "\n".join(lines)


def create_structure_extraction_agent() -> DataScienceAgent:
    """Create a DataScienceAgent configured for structure extraction."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    return DataScienceAgent(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        model="nvidia/nemotron-3-nano-30b-a3b",
        tools=[execute_python_code_tool],  # Only Python tool
        verbose=True,
        stream=True,
        skip_final_response=True,  # Return raw output directly
        max_iterations=5
    )


def main():
    # Data directory
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "context")

    # Initialize agent
    agent = create_structure_extraction_agent()

    # Extract all structures
    print(f"Extracting structures from: {data_dir}")
    structures = extract_all_structures(agent, data_dir)

    # Print formatted output
    print("\n" + "="*70)
    print("EXTRACTED STRUCTURES (formatted for prompt)")
    print("="*70)
    print(format_structures_for_prompt(structures))

    # Print raw JSON
    print("\n" + "="*70)
    print("EXTRACTED STRUCTURES (raw JSON)")
    print("="*70)
    print(json.dumps(structures, indent=2, default=str))

    # Save to file
    output_path = os.path.join(data_dir, "file_structures.json")
    with open(output_path, 'w') as f:
        json.dump(structures, indent=2, fp=f, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
