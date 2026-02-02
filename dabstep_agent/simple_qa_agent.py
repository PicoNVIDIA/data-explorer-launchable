"""Simple QA Agent: Single-phase solver using DataScienceAgent."""

import json
import os
from typing import Optional, List
from agent import DataScienceAgent
from tools import execute_python_code_tool, read_doc_tool, reset_execution_environment


class SimpleQAAgent:
    """A minimal QA agent that directly solves questions using code execution."""

    SYSTEM_PROMPT = """You are a data science expert. The task is tabular data QA.

Use execute_python_code tool to write code and answer the question.
1. Try to write a solution in one shot
2. Only explore when is error
3. when exploring, don't print the entire dataframe or object, print the header or sample
4. Don't over explore. Once the answer is ready, stop calling tools and reply with the final answer
5. The final answer should be direct and short, exactly as the GUIDELINES. Don't add additional analysis

Output Format:
{
    "agent_answer": your_answer based on provided GUIDELINES
}
Don't write anything else!
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
        self.model = "aws/anthropic/claude-haiku-4-5-v1"
        #"azure/openai/gpt-4o-mini" 
        #"azure/openai/o4-mini"
        #"nvidia/openai/gpt-oss-20b"
        #"openai/openai/gpt-5-mini"

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

        with open('new_solutions.md') as f:
            examples = f.read()
        

        agent = DataScienceAgent(
            base_url=self.base_url,
            api_key=self.api_key,
            max_iterations=self.max_iterations,
            model=self.model,
            verbose=self.verbose,
            tools=[execute_python_code_tool],
            system_prompt=self.SYSTEM_PROMPT,
        )
        agent.reset_conversation()

        prompt = f"""You are analyzing payment transaction data.

{self._get_data_files_info()}

A helper python script is provided. "helper.py". Import functions from it in your code if needed.
Carefully read the examples below
Examples:
{examples}

QUESTION: {question['question']}

GUIDELINES: {question.get('guidelines', 'N/A')}

"""

        return agent.process_prompt(prompt)
