#!/usr/bin/env python3
"""
Helper utilities for the DataScienceAgent.

This module contains utility functions for code manipulation and context building.
"""

import json
from typing import Dict, Any, Union


def ensure_print_in_code(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure the last line of code has print() if it's an expression.

    Args:
        arguments: Tool arguments containing 'code' key

    Returns:
        Modified arguments with print() added if needed
    """
    code = arguments.get("code", "")
    if not code.strip():
        return arguments

    # Split into lines and find the last non-empty, non-comment line
    lines = code.split("\n")
    last_line_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i].strip()
        if line and not line.startswith("#"):
            last_line_idx = i
            break

    if last_line_idx == -1:
        return arguments

    last_line = lines[last_line_idx].strip()

    # Check if already has print()
    if last_line.startswith("print(") or last_line.startswith("print "):
        return arguments

    # Keywords that indicate the line is not a simple expression
    statement_keywords = [
        "if ", "elif ", "else:", "for ", "while ", "def ", "class ",
        "import ", "from ", "with ", "try:", "except ", "finally:",
        "return ", "yield ", "raise ", "assert ", "del ", "pass",
        "break", "continue", "global ", "nonlocal "
    ]

    # Check if the last line starts with any statement keyword
    is_statement = any(last_line.startswith(kw) for kw in statement_keywords)

    # Check if it's an assignment (but not a comparison)
    is_assignment = "=" in last_line and not any(op in last_line for op in ["==", "!=", "<=", ">="])

    # Functions/methods that should NOT be wrapped in print()
    # (they either return None or have side effects like display/save)
    no_print_patterns = [
        ".show(", ".savefig(", ".close(",
        "plt.show(", "plt.savefig(", "plt.close(",
        "display(",
        ".to_csv(", ".to_excel(", ".to_json(", ".to_parquet(",
    ]

    # Check if the last line contains any no-print pattern
    should_skip = any(pattern in last_line for pattern in no_print_patterns)

    # If it's not a statement or assignment, it's likely an expression - wrap in print()
    if not is_statement and not is_assignment and not should_skip:
        # Get the indentation of the last line
        indent = len(lines[last_line_idx]) - len(lines[last_line_idx].lstrip())
        indent_str = lines[last_line_idx][:indent]

        # Replace the last line with print(last_line)
        lines[last_line_idx] = f"{indent_str}print({last_line})"

        # Update the arguments
        arguments = arguments.copy()
        arguments["code"] = "\n".join(lines)

    return arguments


def get_dataframe_context(dataframes: Dict[str, Dict[str, Any]]) -> str:
    """
    Build a context message about currently available dataframes.

    Args:
        dataframes: Dictionary mapping variable names to dataframe info

    Returns:
        Context string to inject into the system prompt
    """
    if not dataframes:
        return ""

    context_lines = ["\n\n=== EXECUTION ENVIRONMENT STATE ==="]
    context_lines.append("The following DataFrames are already loaded and available for use:")

    for var_name, info in dataframes.items():
        shape = info.get("shape", "unknown")
        columns = info.get("columns", [])
        memory_mb = info.get("memory_usage_mb", 0)

        context_lines.append(f"\n• Variable '{var_name}':")
        context_lines.append(f"  - Shape: {shape[0]:,} rows × {shape[1]} columns" if isinstance(shape, tuple) else f"  - Shape: {shape}")
        if columns and columns != "error_getting_columns":
            col_preview = ", ".join(columns[:5])
            if len(columns) > 5:
                col_preview += f", ... ({len(columns)} total)"
            context_lines.append(f"  - Columns: {col_preview}")
        context_lines.append(f"  - Memory: {memory_mb} MB")

    context_lines.append("\nYou can directly use these variables in your code without reloading the data.")
    context_lines.append("=" * 35)

    return "\n".join(context_lines)


def truncate_output(data: Union[str, Dict, Any], max_length: int = 300) -> str:
    """
    Truncate output to a maximum length for display purposes.

    Args:
        data: The data to truncate (string, dict, or any object)
        max_length: Maximum length before truncation (default: 300)

    Returns:
        Truncated string with "... (truncated)" suffix if needed
    """
    # Convert to string representation
    if isinstance(data, dict):
        output = json.dumps(data, indent=2)
    elif isinstance(data, str):
        output = data
    else:
        output = str(data)

    # Truncate if too long
    if len(output) > max_length:
        return output[:max_length] + "... (truncated)"

    return output


def clean_failed_executions(messages: list) -> list:
    """
    Remove failed tool executions from message history.

    This function removes assistant+tool message pairs where the tool
    execution failed (success: false). This keeps the conversation context
    clean and prevents the LLM from being confused by failed attempts.

    Args:
        messages: List of conversation messages

    Returns:
        Cleaned list of messages with failed executions removed
    """
    cleaned_messages = []
    i = 0

    while i < len(messages):
        message = messages[i]

        # Check if this is an assistant message with tool calls
        if message.get("role") == "assistant" and message.get("tool_calls"):
            # Look ahead to find corresponding tool result messages
            j = i + 1
            tool_results = []

            # Collect all tool result messages that follow this assistant message
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_results.append(messages[j])
                j += 1

            # Check if all tool results failed
            all_failed = True
            for tool_msg in tool_results:
                try:
                    result = json.loads(tool_msg.get("content", "{}"))
                    if result.get("success", False):
                        all_failed = False
                        break
                except (json.JSONDecodeError, AttributeError):
                    # If we can't parse, assume it didn't fail
                    all_failed = False
                    break

            # If all tools failed, skip this assistant message and its tool results
            if all_failed and tool_results:
                # Skip the assistant message and all its tool results
                i = j
                continue
            else:
                # Keep the assistant message and its tool results
                cleaned_messages.append(message)
                for tool_msg in tool_results:
                    cleaned_messages.append(tool_msg)
                i = j
                continue

        # Keep all non-assistant or assistant messages without tool calls
        cleaned_messages.append(message)
        i += 1

    return cleaned_messages


def remove_last_successful_execution(messages: list) -> list:
    """
    Remove the most recent successful tool execution from message history.

    After the LLM has processed a successful execution and provided its
    response, we remove the tool call and result to keep context clean.
    This prevents the LLM from repeating information it has already processed.

    Args:
        messages: List of conversation messages

    Returns:
        Cleaned list with the last successful execution removed
    """
    # Work backwards to find the most recent assistant message with tool calls
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]

        # Found an assistant message with tool calls
        if message.get("role") == "assistant" and message.get("tool_calls"):
            # Look ahead to find corresponding tool result messages
            j = i + 1
            tool_result_indices = []

            # Collect indices of all tool result messages
            while j < len(messages) and messages[j].get("role") == "tool":
                tool_result_indices.append(j)
                j += 1

            # Check if any tool succeeded
            has_success = False
            for idx in tool_result_indices:
                try:
                    result = json.loads(messages[idx].get("content", "{}"))
                    if result.get("success", False):
                        has_success = True
                        break
                except (json.JSONDecodeError, AttributeError):
                    pass

            # If we found a successful execution, remove it
            if has_success:
                # Create new list without the assistant message and its tool results
                cleaned = messages[:i] + messages[j:]
                return cleaned

    # No successful execution found, return original
    return messages


def calculate_context_tokens(messages: list) -> Dict[str, Any]:
    """
    Calculate the approximate number of tokens in the message context.

    This function counts tokens across all messages in the conversation history,
    including system messages, user prompts, assistant responses, and tool calls.

    Args:
        messages: List of conversation messages

    Returns:
        Dictionary containing:
            - total_tokens: Total estimated tokens in context
            - message_count: Number of messages in context
            - breakdown_tokens: Token count by message role
            - char_count: Total character count
    """
    total_chars = 0
    breakdown = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool": 0
    }

    for message in messages:
        role = message.get("role", "unknown")

        # Count content tokens
        content = message.get("content", "")
        if content:
            char_count = len(str(content))
            total_chars += char_count
            if role in breakdown:
                breakdown[role] += char_count

        # Count tool call tokens (for assistant messages)
        if message.get("tool_calls"):
            tool_calls_str = json.dumps(message["tool_calls"])
            char_count = len(tool_calls_str)
            total_chars += char_count
            breakdown["assistant"] += char_count

    # Rough approximation: 1 token ≈ 4 characters for English text
    # This is a conservative estimate that works across different tokenizers
    total_tokens = total_chars // 4

    return {
        "total_tokens": total_tokens,
        "message_count": len(messages),
        "breakdown_chars": breakdown,
        "breakdown_tokens": {role: chars // 4 for role, chars in breakdown.items()},
        "char_count": total_chars
    }


def clean_assistant_content(content: str) -> str:
    """Strip hidden planner segments and protocol tokens from assistant output."""
    if not content:
        return content

    cleaned = content
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    for sentinel in ("<|im_start|>", "<|im_end|>"):
        if sentinel in cleaned:
            cleaned = cleaned.split(sentinel, 1)[0]
    return cleaned.strip()
