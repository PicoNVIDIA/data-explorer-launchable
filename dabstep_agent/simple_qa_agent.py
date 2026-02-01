"""Simple QA Agent: Single-phase solver using DataScienceAgent."""

import json
import os
from typing import Optional, List
from agent import DataScienceAgent
from tools import execute_python_code_tool, read_doc_tool, reset_execution_environment


class SimpleQAAgent:
    """A minimal QA agent that directly solves questions using code execution."""

    SYSTEM_PROMPT = """You are a data science expert.
Write complete, executable Python code to answer the question.
IMPORTANT: Do NOT assume variables exist. Always import libraries and load data files.
Use pandas to explore tabular data.
Use print() to show results. Preserve exact case of data values.
If you need guidance on fee matching logic, use read_doc to read "./dabstep_agent/fee_matching_guide.md".
"""

    def __init__(
        self,
        data_dir: str = "data/context",
        tasks_file: str = "data/tasks_dev.json",
        file_structures: Optional[str] = None,
        max_iterations: int = 10,
        verbose: bool = True,
    ):
        self.data_dir = data_dir
        self.tasks_file = tasks_file
        self.file_structures = file_structures
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.api_key = os.environ.get("NV_INFER")
        self.base_url = "https://inference-api.nvidia.com"
        self.model = "openai/openai/gpt-5-mini"

        with open(tasks_file, 'r') as f:
            self._questions = json.load(f)

    def _get_data_files_info(self) -> str:
        if self.file_structures:
            return f"Available data files in '{self.data_dir}/':\n{self.file_structures}"
        return f"Data directory: {self.data_dir}/ (scan for csv/json/jsonl files)"

    def get_question(self, index: int) -> dict:
        return self._questions[index]

    def solve(self, question_id: int) -> str:
        """Solve a question by index."""
        reset_execution_environment()
        question = self.get_question(question_id)

        with open('solutions.md') as f:
            examples = f.read()
        

        agent = DataScienceAgent(
            base_url=self.base_url,
            api_key=self.api_key,
            max_iterations=self.max_iterations,
            model=self.model,
            verbose=self.verbose,
            tools=[execute_python_code_tool, read_doc_tool],
            system_prompt=self.SYSTEM_PROMPT+'\nExamples\n'+examples,
        )
        agent.reset_conversation()

        prompt = f"""You are analyzing payment transaction data.

{self._get_data_files_info()}

QUESTION: {question['question']}

GUIDELINES: {question.get('guidelines', 'N/A')}

Use execute_python_code to analyze the data and answer the question."""

        return agent.process_prompt(prompt)
