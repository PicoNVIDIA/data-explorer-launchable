import json
from typing import Optional, Dict
from extract_structure_example import extract_all_structures, format_structures_for_prompt


def load_question(json_path: str, index: int = 0) -> dict:
    """Load a specific question from a JSON file."""
    with open(json_path, 'r') as f:
        questions = json.load(f)
    if index < 0 or index >= len(questions):
        raise IndexError(f"Question at index {index} not found")
    return questions[index]


def solve_question(agent, question_data: dict, data_dir: str = "data", file_structures: Optional[str] = None) -> str:
    """
    Use DataScienceAgent to solve a DABStep benchmark question.

    Args:
        question_data: Dictionary containing task_id, question, guidelines, etc.
        data_dir: Path to the data directory containing CSV/JSON files
        file_structures: Pre-extracted file structures formatted for prompt injection.
                        If None, uses default hardcoded descriptions.

    Returns:
        The agent's answer to the question
    """
    # Use extracted structures if provided, otherwise use default descriptions
    if file_structures:
        data_files_section = f"Available data files in '{data_dir}/':\n{file_structures}"
    else:
        data_files_section = f"""Available data files in '{data_dir}/':
- payments.csv: Payment transactions with columns including psp_reference, merchant, card_scheme,
  year, hour_of_day, eur_amount, ip_country, issuing_country, device_type, has_fraudulent_dispute, etc.
- fees.json: Fee structures
- merchant_data.json: Merchant information
- acquirer_countries.csv: Acquirer country data
- merchant_category_codes.csv: MCC codes"""

    # Construct the prompt with context about available data
    prompt = f"""You are analyzing payment transaction data for a data science benchmark.

{data_files_section}

IMPORTANT DOCUMENTATION:
- manual.md: Contains detailed definitions of terms, concepts, column descriptions, and business rules.
  Some of these terms have different definition than common sense.
  If you need to understand a term, use the search_doc tool:
  - search_doc(term="term_name", file_path="{data_dir}/manual.md")
  The tool will automatically search and return the most relevant passages.

QUESTION: {question_data['question']}

GUIDELINES: {question_data['guidelines']}

INSTRUCTIONS:
1. First, identify the terms in the question, use search_doc to look up their definitions in manual.md
2. Then use execute_python_code to load and analyze the relevant data files
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
    Solve a DABStep benchmark question.

    Args:
        agent: The main DataScienceAgent for solving questions
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
    print("DABStep Benchmark - Solving Question")
    print("=" * 70 + "\n")

    # Solve the question
    answer = solve_question(agent, question, data_dir, file_structures)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Question: {question['question']}")
    print(f"Agent's Answer: {answer}")
    print(f"Expected Answer: {question.get('answer', 'N/A')}")
    print("=" * 70)

    return answer