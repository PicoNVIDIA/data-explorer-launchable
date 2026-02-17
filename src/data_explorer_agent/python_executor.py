"""
Stateful Python code execution tool for NAT.

Ports dabstep_agent/tools.py execute_python_code() to a NAT function group.
Uses exec() with a persistent namespace dict — no Jupyter notebooks.
"""

import io
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


class PythonExecutorConfig(FunctionGroupBaseConfig, name="python_executor"):
    """Stateful Python executor using exec() with a persistent namespace."""
    timeout: int = Field(default=120, description="Max execution time in seconds per code block.")


@register_function_group(config_type=PythonExecutorConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def python_executor(config: PythonExecutorConfig, builder: Builder):
    group = FunctionGroup(config=config)

    # ---- persistent state lives here, in the closure ----
    import pandas as pd
    namespace: dict = {"pd": pd}

    call_count = {"n": 0}

    async def execute_python_code(code: str) -> str:
        """Execute Python code. Variables persist between calls so you can
        build on previous results without reloading data."""
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
                exec(code, namespace)
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

    async def reset_environment(unused: str = "") -> str:
        """Clear all variables and start fresh."""
        print(f"\n{'='*60}")
        print(f"[Tool Call] reset_environment")
        print(f"{'='*60}\n")
        namespace.clear()
        namespace["pd"] = pd
        result = "Environment reset. Only pandas (pd) is available."
        print(f"OUTPUT: {result}\n")
        return result

    group.add_function(
        name="execute_python_code",
        fn=execute_python_code,
        description=execute_python_code.__doc__,
    )
    group.add_function(
        name="reset_environment",
        fn=reset_environment,
        description=reset_environment.__doc__,
    )

    yield group
