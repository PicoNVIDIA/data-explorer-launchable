import json
import os
from typing import Optional, Dict
from extract_structure_example import extract_all_structures, format_structures_for_prompt
from agent import DataScienceAgent
from tools import execute_python_code_tool, search_doc_tool


def load_question(json_path: str, index: int = 0) -> dict:
    """Load a specific question from a JSON file."""
    with open(json_path, 'r') as f:
        questions = json.load(f)
    if index < 0 or index >= len(questions):
        raise IndexError(f"Question at index {index} not found")
    return questions[index]


RESEARCH_SYSTEM_PROMPT = """/no_think You are a documentation researcher. Your task is to search for domain-specific term definitions.
Use search_doc(term="term", file_path="path") to look up terms.
Search for 1-3 key terms that may have special definitions. Do NOT search common words.
After searching, provide a VERY BRIEF summary (max 50 words, just use bullet points, no intro, no headers, no conclusions).  """

SOLVER_SYSTEM_PROMPT = """/no_think You are a data science expert.
Write complete, executable Python code to answer the question.
IMPORTANT: Do NOT assume variables exist. Always import libraries and load data files.
Use pandas to explore tabular data.
Use print() to show results. Preserve exact case of data values."""


def create_research_agent(data_dir: str = "data/context") -> DataScienceAgent:
    """Create an agent with only the search_doc tool for researching documentation."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    agent = DataScienceAgent(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        max_iterations=5,
        model="nvidia/nemotron-3-nano-30b-a3b",
        verbose=True,
        stream=True,
        skip_final_response=False,
        tools=[search_doc_tool],
        system_prompt=RESEARCH_SYSTEM_PROMPT
    )
    agent.reset_conversation()
    return agent


def create_solver_agent() -> DataScienceAgent:
    """Create an agent with only the execute_python_code tool for solving questions."""
    api_key = os.environ.get("NVIDIA_API_KEY")
    agent = DataScienceAgent(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        max_iterations=100,
        model="nvidia/nemotron-3-nano-30b-a3b",
        verbose=True,
        stream=True,
        skip_final_response=False,
        tools=[execute_python_code_tool],
        system_prompt=SOLVER_SYSTEM_PROMPT
    )
    agent.reset_conversation()
    return agent


def research_question(agent: DataScienceAgent, question_data: dict, data_dir: str = "data") -> str:
    """
    Use the research agent to find relevant information from documentation.

    Args:
        agent: Research agent with search_doc tool
        question_data: Dictionary containing the question
        data_dir: Path to the data directory

    Returns:
        String containing the relevant information found.
    """
    prompt = f"""QUESTION: {question_data['question']}

File path: {data_dir}/manual.md"""

    response = agent.process_prompt(prompt)
    return response


def solve_question(agent: DataScienceAgent, question_data: dict, data_dir: str = "data",
                   file_structures: Optional[str] = None,
                   research_info: Optional[str] = None) -> str:
    """
    Use DataScienceAgent to solve a DABStep benchmark question.

    Args:
        agent: The solver agent (should have execute_python_code tool)
        question_data: Dictionary containing task_id, question, guidelines, etc.
        data_dir: Path to the data directory containing CSV/JSON files
        file_structures: Pre-extracted file structures formatted for prompt injection.
        research_info: Information gathered by the research agent about relevant terms.

    Returns:
        The agent's answer to the question
    """
    # Use extracted structures if provided, otherwise use default descriptions
    if file_structures:
        data_files_section = f"Available data files in '{data_dir}/':\n{file_structures}"
    else:
        data_files_section = f"""Available data files in '{data_dir}/':
- payments.csv: Payment transactions
- fees.json: Fee structures
- merchant_data.json: Merchant information
- acquirer_countries.csv: Acquirer country data
- merchant_category_codes.csv: MCC codes"""

    # Build research info section if available
    research_section = ""
    if research_info:
        research_section = f"""
RELEVANT DOCUMENTATION (from research phase):
{research_info}
"""

    # Construct the prompt with context about available data
    prompt = f"""You are analyzing payment transaction data for a data science benchmark.

{data_files_section}
{research_section}
QUESTION: {question_data['question']}

GUIDELINES: {question_data['guidelines']}

INSTRUCTIONS:
1. Use the provided documentation information above to understand the terms and definitions
2. Use execute_python_code to load and analyze the relevant data files
3. Provide the final answer following the guidelines exactly
"""

    print("=" * 70)
    print(f"Task ID: {question_data['task_id']}")
    print(f"Question: {question_data['question']}")
    print(f"Level: {question_data['level']}")
    print(f"Expected Answer: {question_data.get('answer', 'N/A')}")
    print("=" * 70)

    # Get the agent's response
    response = agent.process_prompt(prompt)

    return response

def solver(agent, question_id, file_structures: Optional[str] = None, structure_agent=None):
    """
    Solve a DABStep benchmark question using a two-agent approach.

    Phase 1: Research agent searches documentation for relevant terms and definitions
    Phase 2: Solver agent uses the research results to solve the question with Python code

    Args:
        agent: Ignored (kept for backwards compatibility). Agents are created internally.
        question_id: Index of the question to solve
        file_structures: Pre-extracted file structures. If None and structure_agent
                        is provided, structures will be extracted automatically.
        structure_agent: Agent to use for extracting file structures (optional)

    Returns:
        The agent's answer
    """
    data_dir = "data/context"
    dev_jsonl = "data/tasks_dev.json"

    # Extract file structures if not provided but structure_agent is available
    if file_structures is None and structure_agent is not None:
        print("Extracting file structures...")
        structures = extract_all_structures(structure_agent, data_dir)
        file_structures = format_structures_for_prompt(structures)
        print("File structures extracted.\n")

    # Load the question
    question = load_question(dev_jsonl, index=question_id)

    print("\n" + "=" * 70)
    print("DABStep Benchmark - Solving Question (Two-Agent Approach)")
    print("=" * 70 + "\n")

    # Phase 1: Research - Find relevant documentation
    print("\n" + "=" * 70)
    print("PHASE 1: RESEARCH - Finding relevant documentation")
    print("=" * 70 + "\n")

    research_agent = create_research_agent()
    research_info = research_question(research_agent, question, data_dir)

    # Filter out error/warning messages from research results
    if research_info and "Warning: Reached maximum iterations" in research_info:
        research_info = None

    print("\n" + "=" * 70)
    print("RESEARCH RESULTS:")
    print("-" * 70)
    print(research_info if research_info else "(No results)")
    print("=" * 70 + "\n")

    # Phase 2: Solve - Analyze data with Python
    print("\n" + "=" * 70)
    print("PHASE 2: SOLVING - Analyzing data with Python")
    print("=" * 70 + "\n")

    solver_agent = create_solver_agent()
    answer = solve_question(solver_agent, question, data_dir, file_structures, research_info)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Question: {question['question']}")
    print(f"Agent's Answer: {answer}")
    print(f"Expected Answer: {question.get('answer', 'N/A')}")
    print("=" * 70)

    return answer