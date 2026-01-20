"""
Data Search Sub-Agent: Reuses DataScienceAgent to search non-markdown data files.
Uses Python to query structured data (JSON, CSV) and find relevant information.

Workflow:
1. Identify search terms from the question
2. Use grep to find which files contain each term (no LLM)
3. Rephrase query with relevant file paths and run agent
"""

import os
import json
import subprocess
from typing import List, Dict, Set
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def extract_search_terms(question: str, verbose: bool = False) -> List[str]:
    """
    Use LLM to extract search terms from a question.
    The LLM identifies key terms that should be searched for in data files.

    Args:
        question: The user's question
        verbose: Print debug info

    Returns:
        List of search terms to grep for
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get("NVIDIA_API_KEY")
    )

    prompt = f"""Extract search terms from this question that should be used to find relevant data files.

Question: {question}

Return ONLY a JSON array of search terms. Include:
- Specific names (e.g., company names, product names)
- Field or column names mentioned (e.g., "product type", "category")
- Key phrases that might appear in data files

Example:
Question: "What is the price for product type B from vendor Acme Corp in the Electronics category?"
Output: ["Acme Corp", "product type", "Electronics"]

JSON array:"""

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-30b-a3b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        extra_body={"reasoning_budget":16384,"chat_template_kwargs":{"enable_thinking":False}},
        max_tokens=200
    )
    print(response)
    response_text = response.choices[0].message.content.strip()

    if verbose:
        print(f"LLM response: {response_text}")

    # Parse JSON array from response
    try:
        # Try to extract JSON array from response
        if '[' in response_text:
            start = response_text.index('[')
            end = response_text.rindex(']') + 1
            terms = json.loads(response_text[start:end])
        else:
            terms = []
    except (json.JSONDecodeError, ValueError) as e:
        if verbose:
            print(f"Failed to parse terms: {e}")
        terms = []

    # Filter out empty or very short terms
    terms = [t.strip() for t in terms if isinstance(t, str) and len(t.strip()) > 0]

    return terms


def grep_files_for_term(term: str, data_dir: str) -> List[str]:
    """
    Use grep to find which files in data_dir contain the term.
    No LLM involved - pure grep search.

    Args:
        term: The term to search for
        data_dir: Directory to search in

    Returns:
        List of file paths that contain the term
    """
    try:
        # Use grep -l to list files containing the term
        # -r for recursive, -i for case-insensitive, -l for filenames only
        result = subprocess.run(
            ['grep', '-ril', term, data_dir],
            capture_output=True,
            text=True,
            timeout=30
        )
        files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return files
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"Grep error for term '{term}': {e}")
        return []


def identify_relevant_files(question: str, data_dir: str = "data/context", verbose: bool = False) -> Dict[str, List[str]]:
    """
    Identify which files contain terms from the question using grep.

    Args:
        question: The user's question
        data_dir: Directory containing data files
        verbose: Print debug info

    Returns:
        Dictionary mapping terms to list of files containing them
    """
    terms = extract_search_terms(question, verbose=verbose)
    term_to_files = {}

    for term in terms:
        files = grep_files_for_term(term, data_dir)
        if files:
            term_to_files[term] = files

    return term_to_files


def get_all_relevant_files(term_to_files: Dict[str, List[str]]) -> Set[str]:
    """Get unique set of all relevant files from term mapping."""
    all_files = set()
    for files in term_to_files.values():
        all_files.update(files)
    return all_files


def search_data_files(question: str, data_dir: str = "data/context", verbose: bool = True) -> dict:
    """
    Search data files to find information relevant to the question.

    Steps:
    1. Extract search terms from question
    2. Use grep to find which files contain each term
    3. Build focused prompt with relevant file paths
    4. Run agent to query the data

    Args:
        question: The question to answer
        data_dir: Directory containing data files
        verbose: Print debug info

    Returns:
        Dictionary with findings
    """
    # Step 1 & 2: Identify terms and grep for relevant files
    if verbose:
        print("=" * 70)
        print("Step 1: Extracting search terms...")

    term_to_files = identify_relevant_files(question, data_dir, verbose=verbose)

    if verbose:
        print(f"Found terms: {list(term_to_files.keys())}")
        for term, files in term_to_files.items():
            print(f"  '{term}' -> {files}")
    1/0
    # Get all unique relevant files
    relevant_files = get_all_relevant_files(term_to_files)

    if verbose:
        print(f"\nRelevant files: {relevant_files}")
        print("=" * 70)

    if not relevant_files:
        # Fallback: list all data files if no matches
        relevant_files = set()
        for f in os.listdir(data_dir):
            if f.endswith(('.json', '.csv')):
                relevant_files.add(os.path.join(data_dir, f))

    # Step 3: Create the agent
    agent = DataScienceAgent(
        api_key=os.environ.get("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/nemotron-3-nano-30b-a3b",
        max_iterations=100,
        verbose=verbose,
        stream=True,
        skip_final_response=False,
        tools=[get_file_header_tool, execute_python_code_tool]
    )
    agent.reset_conversation()

    # Step 4: Build focused prompt like test_agent.py
    files_list = '\n'.join(f"  - {f}" for f in sorted(relevant_files))

    prompt = f"""You have access to the following data files:
{files_list}

Question: {question}

Write Python code to load the relevant files and answer the question.
"""

    if verbose:
        print("\nStep 2: Running agent with prompt...")
        print(prompt)
        print("=" * 70)

    response = agent.process_prompt(prompt)

    return {
        "success": True,
        "terms_found": term_to_files,
        "relevant_files": list(relevant_files),
        "findings": response,
        "data_dir": data_dir
    }


# Tool definition for main agent
data_search_tool = {
    "type": "function",
    "function": {
        "name": "search_data_files",
        "description": "Search non-markdown data files (CSV, JSON) to find which files contain information relevant to answering a question. Uses Python to query structured data. Returns which files are relevant and key insights.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to find relevant data files for"
                },
                "data_dir": {
                    "type": "string",
                    "description": "Directory containing data files (default: data/context)",
                    "default": "data/context"
                }
            },
            "required": ["question"]
        }
    }
}


if __name__ == "__main__":
    test_question = """For account type H and the MCC description: Eating Places and Restaurants,
    what would be the average fee that the card scheme GlobalCard would charge for a transaction value of 10 EUR?"""

    print("=" * 70)
    print("Testing DataSearchAgent")
    print("=" * 70)
    print(f"Question: {test_question}")
    print("=" * 70)

    result = search_data_files(test_question, verbose=True)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Terms found: {result['terms_found']}")
    print(f"Relevant files: {result['relevant_files']}")
    print(f"\nFindings:\n{result['findings']}")
