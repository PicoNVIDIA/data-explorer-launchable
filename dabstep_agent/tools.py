"""
Tools for AI agent to interact with CSV data and execute pandas code.
Includes both CPU and GPU-accelerated execution options.
"""

import io
import sys
import contextlib
import time
import subprocess
import shlex

from retriever import search_doc, search_doc_tool


# =============================================================================
# PERSISTENT EXECUTION ENVIRONMENT
# =============================================================================

# Flag to track if GPU acceleration has been initialized
_GPU_INITIALIZED = False

# Persistent namespace that carries over between executions
PERSISTENT_NAMESPACE = {}


def _initialize_gpu_pandas():
    """Initialize GPU-accelerated pandas once at module level."""
    global _GPU_INITIALIZED
    if not _GPU_INITIALIZED:
        try:
            import cudf.pandas
            cudf.pandas.install()
            _GPU_INITIALIZED = True
            print("[GPU Acceleration] cudf.pandas initialized successfully")
        except Exception as e:
            print(f"[GPU Acceleration] Warning: Failed to initialize cudf.pandas: {e}")
            print("[GPU Acceleration] Falling back to CPU mode")


def _initialize_namespace():
    """Initialize or reset the persistent namespace with pandas."""
    global PERSISTENT_NAMESPACE
    # Import pandas AFTER cudf.pandas.install() has been called
    import pandas as pd
    import matplotlib.pyplot as plt
    PERSISTENT_NAMESPACE = {'pd': pd, 'plt': plt}


def reset_execution_environment():
    """Reset the persistent execution environment, clearing all variables."""
    global _GPU_INITIALIZED
    _initialize_namespace()
    return {
        "success": True,
        "message": "Execution environment reset. All variables cleared.",
        "gpu_mode": _GPU_INITIALIZED
    }


# Initialize GPU acceleration FIRST, then import pandas
_initialize_gpu_pandas()
# Now initialize the namespace with GPU-accelerated pandas
_initialize_namespace()


# =============================================================================
# TOOL FUNCTIONS
# =============================================================================

def get_file_header(file_path: str, num_lines: int = 10, verbose: bool = True) -> dict:
    """
    Get the first N lines of a file (like the 'head' command).

    Args:
        file_path: Path to the file
        num_lines: Number of lines to return (default: 10)
        verbose: If True, print the output. Default True.

    Returns:
        Dictionary with file header content
    """
    if verbose:
        print(f"\n[Reading first {num_lines} lines of {file_path}]")
        print("-" * 60)

    try:
        lines = []
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= num_lines:
                    break
                lines.append(line.rstrip('\n\r'))

        content = '\n'.join(lines)

        if verbose:
            print(content)
            print("-" * 60)

        return {
            "success": True,
            "file_path": file_path,
            "num_lines": len(lines),
            "content": content
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def execute_python_code(code: str, use_gpu: bool = True, verbose: bool = True) -> dict:
    """
    Execute Python code using pandas with GPU acceleration.

    Variables persist between executions in the PERSISTENT_NAMESPACE,
    allowing dataframes to be reused across multiple calls.

    Note: GPU acceleration via cudf.pandas is initialized at module load time
    (before pandas import) and applies to all executions if available.

    Args:
        code: Python code string to execute
        use_gpu: Kept for API compatibility. GPU is initialized at module load.
        verbose: If True, print execution details and output. Default True.

    Returns:
        Dictionary with execution results, timing, dataframe tracking, and any errors
    """
    global PERSISTENT_NAMESPACE, _GPU_INITIALIZED

    mode = "gpu_accelerated" if _GPU_INITIALIZED else "cpu"

    if verbose:
        print(f"\n[Executing Python Code - {'GPU Accelerated' if _GPU_INITIALIZED else 'CPU'} Mode]")
        print("-" * 60)
        print(code)
        print("-" * 60)

    try:
        start_time = time.time()

        # Track dataframes before execution
        dataframes_before = _get_dataframe_info(PERSISTENT_NAMESPACE)

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, PERSISTENT_NAMESPACE)

        end_time = time.time()
        execution_time = end_time - start_time

        # Track dataframes after execution
        dataframes_after = _get_dataframe_info(PERSISTENT_NAMESPACE)

        # Determine what changed
        dataframe_changes = _compute_dataframe_changes(dataframes_before, dataframes_after)

        # Get captured output
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()

        # If no output but code contains imports, provide feedback
        if not stdout_output.strip() and 'import ' in code:
            stdout_output = f"Libraries imported successfully: {code}\n Don't import them again"

        # Print the output so user can see it (only if verbose)
        if verbose:
            if stdout_output:
                print(stdout_output, end='')
            if stderr_output:
                print(stderr_output, end='', file=sys.stderr)

        return {
            "success": True,
            "mode": mode,
            "execution_time_seconds": round(execution_time, 4),
            "stdout": stdout_output,
            "stderr": stderr_output,
            "dataframes": dataframes_after,
            "dataframe_changes": dataframe_changes,
            "message": f"Code executed successfully on {'GPU' if _GPU_INITIALIZED else 'CPU'} in {execution_time:.4f} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "mode": mode,
            "error": str(e),
            "error_type": type(e).__name__
        }


def execute_bash_command(command: str, timeout: int = 60, verbose: bool = True, max_output_kb: int = 50) -> dict:
    """
    Execute a bash command and return the output.

    Args:
        command: The bash command to execute (e.g., "find . -name '*.py'", "grep -r 'pattern' .")
        timeout: Maximum execution time in seconds (default: 60)
        verbose: If True, print execution details and output. Default True.
        max_output_kb: Maximum output size in KB before failing (default: 50KB)

    Returns:
        Dictionary with command output, return code, and any errors
    """
    if verbose:
        print(f"\n[Executing Bash Command]")
        print("-" * 60)
        print(command)
        print("-" * 60)

    try:
        start_time = time.time()

        # Execute the command using shell=True to support pipes, redirects, etc.
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        end_time = time.time()
        execution_time = end_time - start_time

        stdout_output = result.stdout
        stderr_output = result.stderr

        # Deduplicate output lines while preserving order
        if stdout_output:
            seen = set()
            unique_lines = []
            for line in stdout_output.splitlines():
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)
            stdout_output = '\n'.join(unique_lines)
            if unique_lines:
                stdout_output += '\n'  # Preserve trailing newline

        # Check if output is too large
        max_output_bytes = max_output_kb * 1024
        output_size = len(stdout_output.encode('utf-8')) if stdout_output else 0

        if output_size > max_output_bytes:
            if verbose:
                print(f"[OUTPUT TOO LARGE: {output_size // 1024}KB > {max_output_kb}KB limit]")
            return {
                "success": False,
                "error": f"Output too large ({output_size // 1024}KB exceeds {max_output_kb}KB limit). Use Python with pandas/json to filter data instead of bash commands on large files.",
                "error_type": "OutputTooLarge",
                "output_size_kb": output_size // 1024
            }

        # Print the output so user can see it (only if verbose)
        if verbose:
            if stdout_output:
                print(stdout_output, end='')
            if stderr_output:
                print(stderr_output, end='', file=sys.stderr)

        # For grep, return code 1 means "no matches found" (not an error)
        # Only treat as failure if return code > 1 or if there's stderr output
        is_success = result.returncode == 0 or (result.returncode == 1 and not stderr_output)

        return {
            "success": is_success,
            "return_code": result.returncode,
            "execution_time_seconds": round(execution_time, 4),
            "stdout": stdout_output if stdout_output else "(no matches found)" if result.returncode == 1 else "",
            "stderr": stderr_output,
            "message": f"Command executed in {execution_time:.4f} seconds with return code {result.returncode}"
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "error": f"Command timed out after {timeout} seconds",
            "error_type": "TimeoutExpired"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


def _get_dataframe_info(namespace: dict) -> dict:
    """
    Extract information about pandas DataFrames in the namespace.

    Args:
        namespace: The execution namespace to inspect

    Returns:
        Dictionary mapping variable names to dataframe metadata
    """
    import pandas as pd
    dataframe_info = {}

    for var_name, var_value in namespace.items():
        # Skip built-in items and modules
        if var_name.startswith('_') or var_name == 'pd':
            continue

        # Check if it's a DataFrame
        if isinstance(var_value, pd.DataFrame):
            try:
                dataframe_info[var_name] = {
                    "shape": var_value.shape,
                    "columns": var_value.columns.tolist()[:10],  # Limit to first 10 columns
                    "dtypes": var_value.dtypes.astype(str).to_dict() if len(var_value.columns) <= 20 else "too_many_columns",
                    "memory_usage_mb": round(var_value.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                }
            except Exception:
                # In case of any error getting metadata
                dataframe_info[var_name] = {
                    "shape": var_value.shape,
                    "columns": "error_getting_columns"
                }

    return dataframe_info


def _compute_dataframe_changes(before: dict, after: dict) -> dict:
    """
    Compute what dataframes were added, modified, or removed.

    Args:
        before: Dataframe info before execution
        after: Dataframe info after execution

    Returns:
        Dictionary with added, modified, and removed dataframe names
    """
    before_names = set(before.keys())
    after_names = set(after.keys())

    added = list(after_names - before_names)
    removed = list(before_names - after_names)

    # Check for modifications (shape change)
    modified = []
    for name in before_names & after_names:
        if before[name].get("shape") != after[name].get("shape"):
            modified.append(name)

    return {
        "added": added,
        "modified": modified,
        "removed": removed
    }


# =============================================================================
# TOOL DEFINITIONS FOR OPENAI API
# =============================================================================

get_file_header_tool = {
    "type": "function",
    "function": {
        "name": "get_file_header",
        "description": "Get the first N lines of a file (like the 'head' command). Use this to preview the content and structure of any text file, including CSV, JSON, log files, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file"
                },
                "num_lines": {
                    "type": "integer",
                    "description": "Number of lines to return (default: 10)",
                    "default": 10
                }
            },
            "required": ["file_path"]
        }
    }
}

execute_python_code_tool = {
    "type": "function",
    "function": {
        "name": "execute_python_code",
        "description": "Execute Python pandas code with optional GPU acceleration. By default, uses GPU-accelerated execution via NVIDIA cudf.pandas for better performance. The pandas library (pd) is already imported. IMPORTANT: Variables persist between executions - if you've previously loaded data into a variable like 'df', it will still be available in subsequent calls. This allows you to build on previous work without re-reading data.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The Python pandas code to execute. Use print() to output results. You can reference variables created in previous executions."
                },
                "use_gpu": {
                    "type": "boolean",
                    "description": "If true (default), use GPU acceleration.",
                    "default": True
                }
            },
            "required": ["code"]
        }
    }
}

execute_bash_command_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash_command",
        "description": "Execute a bash shell command and return its output. Useful for file system operations like find, grep, ls, cat, head, tail, wc, etc. Supports pipes, redirects, and complex shell expressions.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute. Examples: 'find . -name \"*.csv\"', 'grep -r \"pattern\" .', 'ls -la', 'head -n 100 file.txt'"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default: 60)",
                    "default": 60
                }
            },
            "required": ["command"]
        }
    }
}


# =============================================================================
# TOOL REGISTRY
# =============================================================================

# Map of tool names to functions
TOOL_FUNCTIONS = {
    "get_file_header": get_file_header,
    "execute_python_code": execute_python_code,
    "execute_bash_command": execute_bash_command,
    "reset_execution_environment": reset_execution_environment,
    "search_doc": search_doc,
}

# List of all tool definitions
ALL_TOOLS = [
    get_file_header_tool,
    execute_python_code_tool,
    execute_bash_command_tool,
    search_doc_tool,
]


def call_tool(tool_name: str, **arguments):
    """
    Execute a tool by name with given arguments.

    Args:
        tool_name: Name of the tool to call
        **arguments: Arguments to pass to the tool

    Returns:
        Tool execution result as dictionary
    """
    if tool_name in TOOL_FUNCTIONS:
        return TOOL_FUNCTIONS[tool_name](**arguments)
    else:
        return {"error": f"Unknown tool: {tool_name}"}
