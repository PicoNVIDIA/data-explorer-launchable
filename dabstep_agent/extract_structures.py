#!/usr/bin/env python3
"""
Extract structure info from CSV and JSON files.
Standalone script that extracts columns and sample values.
"""

import os
import json
from collections import defaultdict
from dotenv import load_dotenv
from extract_structure_example import (
    create_structure_extraction_agent,
    extract_all_structures,
    format_structures_for_prompt
)

load_dotenv()


def find_common_columns(structures: dict) -> dict:
    """Find columns/keys that appear in multiple files.

    Args:
        structures: Dictionary mapping filenames to structure dictionaries

    Returns:
        Dictionary mapping column names to list of files containing them
    """
    column_to_files = defaultdict(list)

    for filename, structure in structures.items():
        if "error" in structure:
            continue

        file_type = structure.get("file_type", "unknown")

        if file_type == "csv":
            columns = structure.get("columns", [])
            for col in columns:
                column_to_files[col].append(filename)
        elif file_type == "json":
            keys = structure.get("keys", [])
            for key in keys:
                column_to_files[key].append(filename)

    # Filter to only columns that appear in multiple files
    common = {col: files for col, files in column_to_files.items() if len(files) > 1}
    return common


def format_common_columns(common_columns: dict) -> str:
    """Format common columns as a graph showing table relationships.

    Args:
        common_columns: Dictionary mapping column names to list of files

    Returns:
        ASCII graph showing table relationships
    """
    if not common_columns:
        return "No common columns found across files."

    # Build edges: (file1, file2) -> [columns]
    from collections import defaultdict
    edges = defaultdict(list)

    for col, files in common_columns.items():
        # Skip index columns
        if col.startswith("Unnamed"):
            continue
        files = sorted(files)
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                edges[(files[i], files[j])].append(col)

    # Build graph output
    lines = []

    # Sort edges by number of shared columns (descending)
    sorted_edges = sorted(edges.items(), key=lambda x: -len(x[1]))

    for (f1, f2), cols in sorted_edges:
        # Shorten filenames
        f1_short = f1.replace('.csv', '').replace('.json', '')
        f2_short = f2.replace('.csv', '').replace('.json', '')
        cols_str = ", ".join(sorted(cols))
        lines.append(f"{f1_short} <--[{cols_str}]--> {f2_short}")

    return "\n".join(lines)


def extract_file_structures(data_dir: str = "data/context") -> str:
    """Extract file structures from all CSV/JSON files in a directory.

    Args:
        data_dir: Directory containing CSV/JSON files

    Returns:
        Formatted string with file structures
    """
    print(f"Extracting file structures from: {data_dir}")
    agent = create_structure_extraction_agent()
    structures = extract_all_structures(agent, data_dir)
    formatted = format_structures_for_prompt(structures)
    print("File structures extracted.\n")
    return formatted


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract CSV/JSON file structures")
    parser.add_argument("--data-dir", type=str, default="data/context",
                        help="Directory containing data files (default: data/context)")
    parser.add_argument("--raw", action="store_true",
                        help="Output raw JSON instead of formatted text")

    args = parser.parse_args()

    agent = create_structure_extraction_agent()
    structures = extract_all_structures(agent, args.data_dir)

    if args.raw:
        print(json.dumps(structures, indent=2, default=str))
    else:
        print(format_structures_for_prompt(structures))

    # Find and display common columns
    print("\n" + "=" * 70)
    print("COMMON COLUMNS (potential table relationships)")
    print("=" * 70)
    common_columns = find_common_columns(structures)
    print(format_common_columns(common_columns))
    print("=" * 70)
