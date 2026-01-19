import json
def load_question(json_path: str, index: int = 0) -> dict:
    """Load a specific question from a JSON file."""
    with open(json_path, 'r') as f:
        questions = json.load(f)
    if index < 0 or index >= len(questions):
        raise IndexError(f"Question at index {index} not found")
    return questions[index]


def solve_question(agent, question_data: dict, data_dir: str = "data") -> str:
    """
    Use DataScienceAgent to solve a DABStep benchmark question.

    Args:
        question_data: Dictionary containing task_id, question, guidelines, etc.
        data_dir: Path to the data directory containing CSV/JSON files

    Returns:
        The agent's answer to the question
    """

    # Construct the prompt with context about available data
    prompt = f"""You are analyzing payment transaction data for a data science benchmark.

Available data files in '{data_dir}/':
- payments.csv: Payment transactions with columns including psp_reference, merchant, card_scheme,
  year, hour_of_day, eur_amount, ip_country, issuing_country, device_type, has_fraudulent_dispute, etc.
- fees.json: Fee structures
- merchant_data.json: Merchant information
- acquirer_countries.csv: Acquirer country data
- merchant_category_codes.csv: MCC codes

IMPORTANT DOCUMENTATION:
- manual.md: Contains detailed definitions of terms, concepts, column descriptions, and business rules.
Some of these terms have different definition than common sense.
  If you need to understand a term, use the execute_bash_command
  tool to search manual.md. For example:
  - `grep -i "term_name" {data_dir}/manual.md` to find definition of a term
  - `grep -A 5 "term_name" {data_dir}/manual.md` to get context around a term

QUESTION: {question_data['question']}

GUIDELINES: {question_data['guidelines']}

INSTRUCTIONS:
1. First, identify the terms in the question, use execute_bash_command to search manual.md
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

def solver(agent, question_id):
    data_dir = "data/context"
    dev_jsonl = "data/tasks_dev.json"
    
    # Load the first question
    question = load_question(dev_jsonl, index=question_id)
    
    print("\n" + "=" * 70)
    print("DABStep Benchmark - Solving First Question")
    print("=" * 70 + "\n")
    
    # Solve the question
    answer = solve_question(agent, question, data_dir)
    
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"Question: {question['question']}")
    print(f"Agent's Answer: {answer}")
    print(f"Expected Answer: {question.get('answer', 'N/A')}")
    print("=" * 70)