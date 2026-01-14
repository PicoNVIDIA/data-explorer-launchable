import logging
import subprocess
import shlex

from pydantic import Field
from nat.builder.workflow_builder import Builder
from nat.builder.function import FunctionGroup
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function, register_function_group
from nat.data_models.function import FunctionBaseConfig, FunctionGroupBaseConfig

from .notebook_util import NotebookManager, VisionAnalyzerConfig
from typing import Literal, Optional, Dict, Any

logger = logging.getLogger(__name__)


class NotebookFunctionGroupConfig(FunctionGroupBaseConfig, name="notebook_function_group"):
    """
    NAT function group for notebook operations.
    """
    notebook_path: str = Field(description="The path to the notebook to use.")
    vision_analyzer: VisionAnalyzerConfig = Field(description="The vision analyzer to use.")

@register_function_group(config_type=NotebookFunctionGroupConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def notebook_function_group(config: NotebookFunctionGroupConfig, builder: Builder):
    """
    Registers a function group (addressable via `notebook_function_group` in the configuration).
    This registration ensures a static mapping of the function group type, `notebook_function_group`, to the `NotebookFunctionGroupConfig` configuration object.

    Args:
        config (NotebookFunctionGroupConfig): The configuration for the function group.
        builder (Builder): The builder object.

    Returns:
        FunctionInfo: The function info object for the function group.
    """

    group = FunctionGroup(config=config)
    manager = NotebookManager(config.notebook_path, config.vision_analyzer)

    # append cell
    async def append_cell(content: str, cell_type: Literal["code", "markdown"]) -> str:
        """
        Append a new cell to the notebook and execute it.
        """
        result = manager.append_cell(content, cell_type)
        if result["success"]:
            return f"""Successfully appended cell at index {result['cell_index']} ({result['cell_type']}).\nTotal cells: {result['total_cells']}.\nOutput: {result['output']}"""
        else:
            return f"""Appended cell failed to run at index {result['cell_index']}.\nTotal cells: {result['total_cells']}.\nError: {result['error']}"""

    # modify last cell
    async def modify_last_cell(content: str, cell_type: Literal["code", "markdown"]) -> str:
        """
        Modify the last cell in the notebook and execute it.
        """
        result = manager.modify_last_cell(content, cell_type)
        if result["success"]:
            return f"""Successfully modified last cell at index {result['cell_index']} ({result['cell_type']}).\nTotal cells: {result['total_cells']}.\nOutput: {result['output']}"""
        else:
            return f"""Last cell is modified but the notebook failed to run.\nTotal cells: {result['total_cells']}.\nError in cell index {result['error_cell_index']}: {result['error']}"""

    # modify specific cell
    async def modify_cell(cell_index: int, content: str, cell_type: Literal["code", "markdown"]) -> str:
        """
        Modify a specific cell in the notebook and execute it.
        """
        result = manager.modify_cell(cell_index, content, cell_type)
        if result["success"]:
            return f"""Successfully modified cell at index {result['cell_index']} ({result['cell_type']}).\nTotal cells: {result['total_cells']}.\nOutput: {result['output']}"""
        else:
            return f"""Cell at index {result['total_cells']} is modified but the notebook failed to run.\nTotal cells: {result['total_cells']}.\nError in cell index {result['error_cell_index']}: {result['error']}"""

    # delete cell
    async def delete_cell(cell_index: int) -> str:
        """
        Delete a cell in the notebook and execute the entire notebook.
        """
        result = manager.delete_cell(cell_index)
        total_cells = len(manager.notebook.cells)
        if result["success"]:
            return f"""Successfully deleted cell at index {cell_index}.\nTotal cells: {total_cells}."""
        else:
            return f"""Cell at index {cell_index} is not deleted.\nTotal cells: {total_cells}."""

    from pydantic import BaseModel
    class GetNotebookSummaryInput(BaseModel):
        unused: str = Field(description="you must provide this field (string), though it isn't used.")
    # get notebook summary
    async def get_notebook_summary(input: GetNotebookSummaryInput) -> str:
        """
        Get a summary of the notebook.
        """
        return manager.get_notebook_summary()

    # Add all functions to the group
    group.add_function(name="append_cell", fn=append_cell, description=append_cell.__doc__)
    group.add_function(name="modify_last_cell", fn=modify_last_cell, description=modify_last_cell.__doc__)
    group.add_function(name="modify_cell", fn=modify_cell, description=modify_cell.__doc__)
    group.add_function(name="delete_cell", fn=delete_cell, description=delete_cell.__doc__)
    group.add_function(name="get_notebook_summary", fn=get_notebook_summary, description=get_notebook_summary.__doc__)

    yield group


class BashFunctionGroupConfig(FunctionGroupBaseConfig, name="bash_function_group"):
    """
    NAT function group for simple bash operations.
    """
    pass


@register_function_group(config_type=BashFunctionGroupConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def bash_function_group(config: BashFunctionGroupConfig, builder: Builder):
    """
    Registers a function group for simple bash operations.
    """
    group = FunctionGroup(config=config)

    ALLOWED_COMMANDS = {"ls", "find", "grep", "cat", "head", "tail", "wc", "pwd", "echo", "file", "stat", "du", "df"}
    BLOCKED_COMMANDS = {"rm", "rmdir", "mv", "cp", "chmod", "chown", "sudo", "su", "dd", "mkfs", "fdisk", "kill", "pkill", "shutdown", "reboot"}

    async def run_bash(command: str) -> str:
        """
        Run a simple bash command. Only read-only commands like ls, find, grep, cat, head, tail, wc, pwd, echo, file, stat, du, df are allowed.
        Destructive commands like rm, mv, cp, chmod, sudo are blocked.
        """
        try:
            # Parse the command to get the base command
            parts = shlex.split(command)
            if not parts:
                return "Error: Empty command"

            base_cmd = parts[0]

            # Check if command is blocked
            if base_cmd in BLOCKED_COMMANDS:
                return f"Error: Command '{base_cmd}' is not allowed for safety reasons."

            # Check if command is in the allowed list
            if base_cmd not in ALLOWED_COMMANDS:
                return f"Error: Command '{base_cmd}' is not in the allowed list. Allowed commands: {', '.join(sorted(ALLOWED_COMMANDS))}"

            # Run the command
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )

            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\nStderr: {result.stderr}"
            if result.returncode != 0:
                output += f"\nReturn code: {result.returncode}"

            return output.strip() if output.strip() else "(No output)"

        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error: {str(e)}"

    group.add_function(name="run_bash", fn=run_bash, description=run_bash.__doc__)

    yield group