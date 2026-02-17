"""
Stateful Python code execution tool for NAT.

Ports dabstep_agent/tools.py execute_python_code() to a NAT function group.
Uses exec() with a persistent namespace dict — no Jupyter notebooks.
"""

import ast
import io
import os
import sys
import contextlib
import time
import logging

from pydantic import Field
from nat.builder.workflow_builder import Builder
from nat.builder.function import FunctionGroup
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.cli.register_workflow import register_function_group
from nat.data_models.function import FunctionGroupBaseConfig

logger = logging.getLogger(__name__)

# Module-level registry so callers can invoke tools directly without an LLM roundtrip.
# Populated when the function group is created by NAT.
_tools: dict[str, callable] = {}


class PythonExecutorConfig(FunctionGroupBaseConfig, name="python_executor"):
    """Stateful Python executor using exec() with a persistent namespace."""
    timeout: int = Field(default=120, description="Max execution time in seconds per code block.")
    sys_paths: list[str] = Field(default_factory=list, description="Extra directories to add to sys.path for imports.")
    workspace_dir: str = Field(default="workspace", description="Directory to save generated code files.")


def _exec_with_auto_print(code: str, namespace: dict) -> None:
    """Execute code with REPL-style auto-print of the last expression.

    If the last statement is a bare expression (not assignment, import, etc.),
    its result is automatically printed — just like a Jupyter cell or Python REPL.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fall back to plain exec and let it raise its own SyntaxError
        exec(code, namespace)
        return

    if not tree.body:
        return

    last = tree.body[-1]

    # Only auto-print bare expressions (e.g. `df.head()`, `x + 1`, `result`).
    # Skip if it's already a print() call — no double-printing.
    if isinstance(last, ast.Expr) and not _is_print_call(last.value):
        # Compile & exec everything except the last statement
        if len(tree.body) > 1:
            head = ast.Module(body=tree.body[:-1], type_ignores=[])
            ast.fix_missing_locations(head)
            exec(compile(head, "<code>", "exec"), namespace)

        # Eval the last expression and print its result (if not None)
        expr = ast.Expression(body=last.value)
        ast.fix_missing_locations(expr)
        result = eval(compile(expr, "<code>", "eval"), namespace)
        if result is not None:
            print(result)
    else:
        exec(compile(tree, "<code>", "exec"), namespace)


def _is_print_call(node: ast.AST) -> bool:
    """Return True if the node is a call to print()."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    )


@register_function_group(config_type=PythonExecutorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def python_executor(config: PythonExecutorConfig, builder: Builder):
    group = FunctionGroup(config=config)

    # ---- add extra import paths ----
    for p in config.sys_paths:
        abs_p = os.path.abspath(p)
        if abs_p not in sys.path:
            sys.path.insert(0, abs_p)

    # ---- persistent state lives here, in the closure ----
    import pandas as pd
    namespace: dict = {"pd": pd}

    call_count = {"n": 0}
    code_history: list[str] = []

    async def execute_python_code(code: str) -> str:
        """Execute Python code. Variables persist between calls so you can
        build on previous results without reloading data."""
        code_history.append(code)
        call_count["n"] += 1
        n = call_count["n"]
        print(f"\n{'='*60}")
        print(f"[Tool Call #{n}] execute_python_code")
        print(f"{'='*60}")
        print(f"INPUT:\n{code}")
        print(f"{'-'*60}")

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        try:
            start = time.time()
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                _exec_with_auto_print(code, namespace)
            elapsed = time.time() - start

            stdout = stdout_capture.getvalue()
            stderr = stderr_capture.getvalue()

            parts = []
            if stdout:
                parts.append(stdout)
            if stderr:
                parts.append(f"[stderr] {stderr}")
            parts.append(f"[executed in {elapsed:.3f}s]")
            result = "\n".join(parts) if parts else "[executed successfully, no output]"
        except Exception as e:
            result = f"Error ({type(e).__name__}): {e}"

        print(f"OUTPUT:\n{result}")
        print(f"{'='*60}\n")
        return result

    async def save_generated_code(task_id: str) -> str:
        """Save all code snippets from this task to workspace. Call after solving."""
        if not code_history:
            return "[no code to save]"
        workspace = config.workspace_dir
        os.makedirs(workspace, exist_ok=True)
        filepath = os.path.join(workspace, f"task{task_id}.py")
        with open(filepath, "w") as f:
            f.write(f"# Generated code for task {task_id}\n")
            f.write(f"# Total code snippets: {len(code_history)}\n\n")
            for i, code in enumerate(code_history, 1):
                f.write(f"# --- Snippet {i} ---\n")
                f.write(code)
                f.write("\n\n")
        return f"Saved {len(code_history)} snippet(s) to {filepath}"

    async def reset_environment(unused: str = "") -> str:
        """Clear all variables and start fresh."""
        print(f"\n{'='*60}")
        print(f"[Tool Call] reset_environment")
        print(f"{'='*60}\n")
        namespace.clear()
        namespace["pd"] = pd
        code_history.clear()
        call_count["n"] = 0
        result = "Environment reset. Only pandas (pd) is available."
        print(f"OUTPUT: {result}\n")
        return result

    group.add_function(
        name="execute_python_code",
        fn=execute_python_code,
        description=execute_python_code.__doc__,
    )
    group.add_function(
        name="save_generated_code",
        fn=save_generated_code,
        description=save_generated_code.__doc__,
    )
    group.add_function(
        name="reset_environment",
        fn=reset_environment,
        description=reset_environment.__doc__,
    )

    # Expose closures so callers can invoke them without an LLM roundtrip.
    _tools["save_generated_code"] = save_generated_code
    _tools["reset_environment"] = reset_environment

    yield group
