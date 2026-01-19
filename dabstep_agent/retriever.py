"""
Retriever tool for semantic search using NVIDIA embeddings.
"""

import os
import subprocess
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Default configuration
DEFAULT_API_KEY = os.environ.get("NVIDIA_API_KEY")
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_TOP_K = 5


def get_embeddings(texts: list, input_type: str, api_key: str = DEFAULT_API_KEY,
                   base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL) -> list:
    """Get embeddings for a list of texts."""
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    response = client.embeddings.create(
        input=texts,
        model=model,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"}
    )
    return [d.embedding for d in response.data]


def search_doc(term: str, file_path: str, top_k: int = DEFAULT_TOP_K, verbose: bool = True) -> dict:
    """
    Search for a term in a document and return the most relevant passages using semantic search.

    This function:
    1. Runs grep to find lines containing the term
    2. Constructs a semantic query "what is the definition of {term}"
    3. Uses embeddings to rank and return the most relevant passages

    Args:
        term: The term to search for in the document.
        file_path: Path to the document file to search.
        top_k: Number of top results to return (default: 5).
        verbose: If True, print execution details.

    Returns:
        Dictionary with success status and the top-k most relevant passages.
    """
    if verbose:
        print(f"\n[Searching Documents]")
        print("-" * 60)
        print(f"Term: {term}")
        print(f"File: {file_path}")
        print(f"Top-k: {top_k}")
        print("-" * 60)

    # Step 0: Validate file exists
    if not os.path.isfile(file_path):
        return {
            "success": False,
            "error": f"File not found: {file_path}"
        }

    # Step 1: Run grep to get relevant lines with context
    # Using -i for case-insensitive, -B 2 -A 2 for context lines
    try:
        result = subprocess.run(
            ["grep", "-i", "-B", "2", "-A", "2", term, file_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        grep_output = result.stdout
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": "Grep command timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Grep failed: {str(e)}"
        }

    # Handle no results from grep
    if not grep_output.strip():
        return {
            "success": False,
            "error": f"No matches found for term '{term}' in {file_path}"
        }

    # Split output by "--" (grep's context separator) or newlines
    # grep -A -B uses "--" to separate match groups
    raw_chunks = grep_output.split("--")

    # Process each chunk: create smaller passages centered on lines containing the term
    # This prevents unrelated context from diluting the semantic signal
    passages = []
    seen = set()
    term_lower = term.lower()

    for chunk in raw_chunks:
        lines = [line.strip() for line in chunk.strip().split("\n") if line.strip()]
        if not lines:
            continue

        # Find lines that actually contain the search term
        term_line_indices = [i for i, line in enumerate(lines) if term_lower in line.lower()]

        if term_line_indices:
            # Create a small passage around each line containing the term (1 line context each side)
            for idx in term_line_indices:
                start = max(0, idx - 1)
                end = min(len(lines), idx + 2)
                passage = " ".join(lines[start:end])
                if passage and passage not in seen:
                    passages.append(passage)
                    seen.add(passage)
        else:
            # Fallback: if somehow no term found, use the whole chunk
            passage = " ".join(lines)
            if passage and passage not in seen:
                passages.append(passage)
                seen.add(passage)

    if not passages:
        return {
            "success": False,
            "error": f"No valid passages extracted for term '{term}'"
        }

    if verbose:
        print(f"Grep found {len(passages)} passage(s)")

    # Step 2: Construct semantic query
    query = f"what is the definition of {term}"

    if verbose:
        print(f"Semantic query: {query}")
        print("-" * 60)

    # Step 3: Semantic search using embeddings
    try:
        query_embedding = np.array(get_embeddings([query], "query")[0])
        passage_embeddings = np.array(get_embeddings(passages, "passage"))

        # Calculate cosine similarity (vectors are normalized)
        scores = np.dot(passage_embeddings, query_embedding)

        # Get top-k indices (but not more than available passages)
        actual_top_k = min(top_k, len(passages))
        top_indices = np.argsort(scores)[::-1][:actual_top_k]

        # Format results
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                "rank": i + 1,
                "score": float(scores[idx]),
                "passage": passages[idx]
            })

        # Create formatted string for display
        formatted_results = []
        for r in results:
            formatted_results.append(f"{r['rank']}. (score: {r['score']:.4f}) {r['passage']}")

        output = "\n".join(formatted_results)

        if verbose:
            print(output)

        return {
            "success": True,
            "results": results,
            "formatted_output": output,
            "message": f"Found top {len(results)} relevant passages for '{term}'"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Error during semantic search: {str(e)}",
            "error_type": type(e).__name__
        }


# Tool definition for OpenAI API
search_doc_tool = {
    "type": "function",
    "function": {
        "name": "search_doc",
        "description": "Search for a term in the documentation and return the most relevant passages. Use this to look up definitions of terms, concepts, column descriptions, and business rules in manual.md.",
        "parameters": {
            "type": "object",
            "properties": {
                "term": {
                    "type": "string",
                    "description": "The term to search for (e.g., 'authorization rate', 'MCC', 'chargeback')"
                },
                "file_path": {
                    "type": "string",
                    "description": "Path to the document file to search"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top results to return (default: 5)",
                    "default": 1
                }
            },
            "required": ["term", "file_path"]
        }
    }
}


if __name__ == "__main__":
    # Simple test
    test_file = "data/context/manual.md"
    test_term = "fraud"

    print("=" * 60)
    print("Testing search_doc function")
    print("=" * 60)

    result = search_doc(term=test_term, file_path=test_file, top_k=3)

    print("\n" + "=" * 60)
    print("Result:")
    print("=" * 60)
    if result["success"]:
        print(f"Success: {result['message']}")
        print(f"\nTop passages:")
        for r in result["results"]:
            print(f"  {r['rank']}. (score: {r['score']:.4f})")
            print(f"     {r['passage'][:100]}...")
    else:
        print(f"Error: {result['error']}")
