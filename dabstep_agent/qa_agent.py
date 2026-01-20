"""
QAAgent: A class-based agent for solving data analysis benchmark questions.

This agent uses a two-phase approach:
1. Research phase: Search documentation for relevant term definitions
2. Solver phase: Execute Python code to analyze data and answer the question

Example usage:
    agent = QAAgent(data_dir="data/context", tasks_file="data/tasks_dev.json")
    answer = agent.solve(question_id=0)
    # or solve by question dict
    answer = agent.solve_question(question_data)
"""

import glob
import json
import os
from typing import Optional, Dict, List, Any
from agent import DataScienceAgent
from tools import execute_python_code_tool, search_doc_tool, reset_execution_environment


class QAAgent:
    """
    A question-answering agent that combines documentation research with
    code execution to solve data analysis benchmark questions.
    """

    RESEARCH_SYSTEM_PROMPT = """/no_think You are a documentation researcher. Your task is to search for domain-specific term definitions.
Use search_doc(term="term", file_path="path") to look up terms.
Search for 1-3 key terms that may have special definitions. Do NOT search common words.
After searching, provide a VERY BRIEF summary (max 50 words, just use bullet points, no intro, no headers, no conclusions)."""

    SOLVER_SYSTEM_PROMPT = """/no_think You are a data science expert.
Write complete, executable Python code to answer the question.
IMPORTANT: Do NOT assume variables exist. Always import libraries and load data files.
Use pandas to explore tabular data.
Use print() to show results. Preserve exact case of data values."""

    def __init__(
        self,
        data_dir: str = "data/context",
        tasks_file: str = "data/tasks_dev.json",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "nvidia/nemotron-3-nano-30b-a3b",
        research_max_iterations: int = 5,
        solver_max_iterations: int = 100,
        verbose: bool = True,
        stream: bool = True,
        file_structures: Optional[str] = None,
    ):
        """
        Initialize the QAAgent.

        Args:
            data_dir: Directory containing data files (CSVs, JSONs, *.md)
            tasks_file: Path to the JSON file containing task questions
            api_key: API key for the LLM service (defaults to NVIDIA_API_KEY env var)
            base_url: Base URL for the LLM API endpoint
            model: The model to use for both research and solver agents
            research_max_iterations: Max iterations for the research agent
            solver_max_iterations: Max iterations for the solver agent
            verbose: Whether to print detailed execution logs
            stream: Whether to stream LLM outputs
            file_structures: Pre-extracted file structures for prompt injection
        """
        self.data_dir = data_dir
        self.tasks_file = tasks_file
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url
        self.model = model
        self.research_max_iterations = research_max_iterations
        self.solver_max_iterations = solver_max_iterations
        self.verbose = verbose
        self.stream = stream
        self.file_structures = file_structures

        # Load questions
        self._questions: List[Dict[str, Any]] = []
        self._load_questions()

        # Find markdown files in data directory
        self._md_files: List[str] = []
        self._find_markdown_files()

        if self.verbose:
            print(f"QAAgent initialized")
            print(f"  Data directory: {self.data_dir}")
            print(f"  Tasks file: {self.tasks_file}")
            print(f"  Number of questions: {len(self._questions)}")
            print(f"  Markdown files: {self._md_files if self._md_files else '(none found)'}")
            print()

    def _find_markdown_files(self):
        """Find all markdown files in the data directory."""
        pattern = os.path.join(self.data_dir, "*.md")
        self._md_files = glob.glob(pattern)
        # Sort for consistent ordering
        self._md_files.sort()

    def _load_questions(self):
        """Load questions from the tasks file."""
        if os.path.exists(self.tasks_file):
            with open(self.tasks_file, 'r') as f:
                self._questions = json.load(f)
        else:
            self._questions = []
            if self.verbose:
                print(f"Warning: Tasks file not found: {self.tasks_file}")

    def _create_research_agent(self) -> DataScienceAgent:
        """Create a research agent with only the search_doc tool."""
        agent = DataScienceAgent(
            base_url=self.base_url,
            api_key=self.api_key,
            max_iterations=self.research_max_iterations,
            model=self.model,
            verbose=self.verbose,
            stream=self.stream,
            skip_final_response=False,
            tools=[search_doc_tool],
            system_prompt=self.RESEARCH_SYSTEM_PROMPT
        )
        agent.reset_conversation()
        return agent

    def _create_solver_agent(self) -> DataScienceAgent:
        """Create a solver agent with only the execute_python_code tool."""
        agent = DataScienceAgent(
            base_url=self.base_url,
            api_key=self.api_key,
            max_iterations=self.solver_max_iterations,
            model=self.model,
            verbose=self.verbose,
            stream=self.stream,
            skip_final_response=False,
            tools=[execute_python_code_tool],
            system_prompt=self.SOLVER_SYSTEM_PROMPT
        )
        agent.reset_conversation()
        return agent

    def get_question(self, index: int) -> Dict[str, Any]:
        """
        Get a question by index.

        Args:
            index: Index of the question (0-based)

        Returns:
            Dictionary containing the question data

        Raises:
            IndexError: If the index is out of range
        """
        if index < 0 or index >= len(self._questions):
            raise IndexError(f"Question index {index} out of range (0-{len(self._questions) - 1})")
        return self._questions[index]

    @property
    def num_questions(self) -> int:
        """Return the total number of questions."""
        return len(self._questions)

    @property
    def has_markdown_files(self) -> bool:
        """Return True if markdown files are available for research."""
        return len(self._md_files) > 0

    def research(self, question_data: Dict[str, Any]) -> Optional[str]:
        """
        Run the research phase to find relevant documentation.

        Args:
            question_data: Dictionary containing the question

        Returns:
            String containing relevant documentation info, or None if not found
        """
        # Skip if no markdown files available
        if not self._md_files:
            if self.verbose:
                print("\n" + "=" * 70)
                print("PHASE 1: RESEARCH - Skipped (no markdown files found)")
                print("=" * 70 + "\n")
            return None

        if self.verbose:
            print("\n" + "=" * 70)
            print("PHASE 1: RESEARCH - Finding relevant documentation")
            print("=" * 70 + "\n")

        research_agent = self._create_research_agent()

        # Build file paths list for all markdown files
        file_paths = "\n".join([f"- {md_file}" for md_file in self._md_files])

        prompt = f"""QUESTION: {question_data['question']}

Available documentation files:
{file_paths}"""

        response = research_agent.process_prompt(prompt)

        # Filter out error/warning messages
        if response and "Warning: Reached maximum iterations" in response:
            response = None

        if self.verbose:
            print("\n" + "=" * 70)
            print("RESEARCH RESULTS:")
            print("-" * 70)
            print(response if response else "(No results)")
            print("=" * 70 + "\n")

        return response

    def solve_with_research(
        self,
        question_data: Dict[str, Any],
        research_info: Optional[str] = None
    ) -> str:
        """
        Run the solver phase to answer the question using Python code.

        Args:
            question_data: Dictionary containing the question, guidelines, etc.
            research_info: Optional documentation info from research phase

        Returns:
            The agent's answer to the question
        """
        if self.verbose:
            print("\n" + "=" * 70)
            print("PHASE 2: SOLVING - Analyzing data with Python")
            print("=" * 70 + "\n")

        solver_agent = self._create_solver_agent()

        # Build data files section
        if self.file_structures:
            data_files_section = f"Available data files in '{self.data_dir}/':\n{self.file_structures}"
        else:
            data_files_section = f"""Available data files in '{self.data_dir}/':
- payments.csv: Payment transactions
- fees.json: Fee structures
- merchant_data.json: Merchant information
- acquirer_countries.csv: Acquirer country data
- merchant_category_codes.csv: MCC codes"""

        # Build research info section
        research_section = ""
        if research_info:
            research_section = f"""
RELEVANT DOCUMENTATION (from research phase):
{research_info}
"""

        prompt = f"""You are analyzing payment transaction data for a data science benchmark.

{data_files_section}
{research_section}
QUESTION: {question_data['question']}

GUIDELINES: {question_data.get('guidelines', 'N/A')}

INSTRUCTIONS:
1. Use the provided documentation information above to understand the terms and definitions
2. Use execute_python_code to load and analyze the relevant data files
3. Provide the final answer following the guidelines exactly
"""

        if self.verbose:
            print("=" * 70)
            print(f"Task ID: {question_data.get('task_id', 'N/A')}")
            print(f"Question: {question_data['question']}")
            print(f"Level: {question_data.get('level', 'N/A')}")
            print(f"Expected Answer: {question_data.get('answer', 'N/A')}")
            print("=" * 70)

        answer = solver_agent.process_prompt(prompt)
        return answer

    def solve_question(
        self,
        question_data: Dict[str, Any],
        skip_research: bool = False
    ) -> str:
        """
        Solve a question using the two-agent approach.

        Args:
            question_data: Dictionary containing the question data
            skip_research: If True, skip the research phase

        Returns:
            The agent's answer to the question
        """
        # Reset execution environment before solving
        reset_execution_environment()

        if self.verbose:
            print("\n" + "=" * 70)
            print("QAAgent - Solving Question (Two-Agent Approach)")
            print("=" * 70 + "\n")

        # Phase 1: Research (skip if no markdown files or explicitly requested)
        research_info = None
        if not skip_research and self.has_markdown_files:
            research_info = self.research(question_data)
        elif self.verbose and not self.has_markdown_files:
            print("Research phase bypassed: no markdown files in data directory")

        # Phase 2: Solve
        answer = self.solve_with_research(question_data, research_info)

        if self.verbose:
            print("\n" + "=" * 70)
            print("RESULT")
            print("=" * 70)
            print(f"Question: {question_data['question']}")
            print(f"Agent's Answer: {answer}")
            print(f"Expected Answer: {question_data.get('answer', 'N/A')}")
            print("=" * 70)

        return answer

    def solve(
        self,
        question_id: int,
        skip_research: bool = False
    ) -> str:
        """
        Solve a question by its index.

        Args:
            question_id: Index of the question to solve (0-based)
            skip_research: If True, skip the research phase

        Returns:
            The agent's answer to the question
        """
        question_data = self.get_question(question_id)
        return self.solve_question(question_data, skip_research=skip_research)

    def set_file_structures(self, file_structures: str):
        """
        Set the pre-extracted file structures for prompt injection.

        Args:
            file_structures: Formatted string describing available data files
        """
        self.file_structures = file_structures


def load_question(json_path: str, index: int = 0) -> dict:
    """
    Load a specific question from a JSON file.

    This is a standalone utility function for backwards compatibility.

    Args:
        json_path: Path to the JSON file containing questions
        index: Index of the question to load

    Returns:
        Dictionary containing the question data
    """
    with open(json_path, 'r') as f:
        questions = json.load(f)
    if index < 0 or index >= len(questions):
        raise IndexError(f"Question at index {index} not found")
    return questions[index]
