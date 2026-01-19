#!/usr/bin/env python3
"""
DataScienceAgent: An AI-powered data science agent that can write and execute pandas code.

This agent uses NVIDIA's LLM API with function calling to:
- Understand user requests about data analysis
- Inspect CSV files and understand their structure
- Generate appropriate pandas code
- Execute code with CPU or GPU acceleration
- Return results and insights

Example usage:
    agent = DataScienceAgent(api_key="your-api-key")
    response = agent.process_prompt("Analyze the sales data in data.csv")
    print(response)
"""

import os
import json
from typing import Optional, Dict, List, Any
from openai import OpenAI
from tools import ALL_TOOLS, call_tool, reset_execution_environment
from agent_helpers import ensure_print_in_code, get_dataframe_context, truncate_output, clean_failed_executions, remove_last_successful_execution, calculate_context_tokens, clean_assistant_content


class DataScienceAgent:
    """
    An autonomous data science agent that can write and execute pandas code.

    The agent maintains conversation state and can handle multiple prompts
    in sequence, building on previous context.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8000/v1",
        model: str = "nemotron",
        max_iterations: int = 10,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        verbose: bool = True,
        force_final_response_after_success: bool = False,
        stream: bool = False,
        skip_final_response: bool = False
    ):
        """
        Initialize the DataScienceAgent.

        Args:
            api_key: API key for the LLM service. For local servers, use "not-needed".
                     For NVIDIA cloud API, provide NGC_API_KEY or set NGC_API_KEY env var.
            base_url: Base URL for the LLM API endpoint.
                     Default: "http://localhost:8000/v1" (local llama.cpp server)
                     For NVIDIA cloud: "https://integrate.api.nvidia.com/v1"
            model: The model to use. Default "nemotron" for local server.
                   For NVIDIA cloud: "nvidia/nvidia-nemotron-nano-9b-v2"
            max_iterations: Maximum number of agent loop iterations per prompt.
            temperature: Sampling temperature for the LLM.
            max_tokens: Maximum tokens in LLM response.
            verbose: Whether to print detailed execution logs.
            force_final_response_after_success: If True, disable tool access
                immediately after a successful execution so the LLM must
                provide a natural-language summary response.
            stream: If True, stream the LLM output token-by-token as it's generated.
                Works for both regular text and tool calls.
            skip_final_response: If True, skip asking LLM for final summary after successful
                tool execution. Returns the raw tool output directly. Note: The agent will
                still retry on failed executions regardless of this setting. Only skips the
                final response after a successful execution. Useful for faster responses when
                you only need the raw execution results.
        """
        # Configuration
        self.base_url = base_url
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.stream = stream
        self.skip_final_response = skip_final_response
        self.force_final_response_after_success = force_final_response_after_success
        self.awaiting_final_response = False

        # Get API key (allow override for local servers)
        self.api_key = api_key or os.environ.get("NGC_API_KEY") or "not-needed"

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        self.sys_prompt = "/no_think " \
        "You are a data science expert. " \
        "Write complete, executable Python code or call provided tools. " \
        "IMPORTANT: Do NOT assume variables exist unless shown in the execution environment state. " \
        "When asked to 'read' or 'load' a file, write the full code including 'import pandas as pd' and 'pd.read_csv()'. " \
        "Always use GPU acceleration (pandas will automatically use GPU). " \
        "Use print() to show results. " \
        "Preserve exact case of data values. Don't change 'apple' to 'Apple'. " \
        "Now answer user's request:\n"
        self.user_prompt=""

        # Conversation state
        self.messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.sys_prompt
            }
        ]

        # Track execution results
        self.last_execution_result: Optional[Dict[str, Any]] = None

        # Track dataframes in the execution environment
        self.dataframes: Dict[str, Dict[str, Any]] = {}

        if self.verbose:
            print("DataScienceAgent initialized")
            print(f"  Base URL: {self.base_url}")
            print(f"  Model: {self.model}")
            print(f"  Available tools: {[tool['function']['name'] for tool in ALL_TOOLS]}")
            print()

    def _process_streaming_response(self, stream_response):
        """
        Process a streaming response from the LLM API.

        Handles both regular text content and tool calls, printing content as it arrives.

        Args:
            stream_response: The streaming response object from the API

        Returns:
            A message object compatible with the regular (non-streaming) response format
        """
        # Accumulators for the complete message
        content_parts = []
        tool_calls_dict = {}
        finish_reason = None
        streaming_started = False

        for chunk in stream_response:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            finish_reason = choice.finish_reason or finish_reason

            # Handle text content
            if delta.content:
                content_parts.append(delta.content)
                if self.verbose:
                    # Print header on first content
                    if not streaming_started:
                        print("  Streaming response: ", end="", flush=True)
                        streaming_started = True
                    print(delta.content, end="", flush=True)

            # Handle tool calls
            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index

                    # Initialize tool call entry if new
                    if idx not in tool_calls_dict:
                        tool_calls_dict[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {
                                "name": "",
                                "arguments": ""
                            },
                            "_streaming_started": False
                        }

                    # Accumulate tool call fields
                    if tool_call_delta.id:
                        tool_calls_dict[idx]["id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        if tool_call_delta.function.name:
                            tool_calls_dict[idx]["function"]["name"] += tool_call_delta.function.name
                            # Print function name as it streams
                            if self.verbose and not tool_calls_dict[idx]["_streaming_started"]:
                                print(f"\n  Calling: {tool_call_delta.function.name}", end="", flush=True)
                                tool_calls_dict[idx]["_streaming_started"] = True

                        if tool_call_delta.function.arguments:
                            # Ensure we're concatenating strings
                            arg_chunk = tool_call_delta.function.arguments
                            if not isinstance(arg_chunk, str):
                                arg_chunk = str(arg_chunk)
                            tool_calls_dict[idx]["function"]["arguments"] += arg_chunk

                            # Stream the arguments as they arrive
                            if self.verbose:
                                print(arg_chunk, end="", flush=True)

        if self.verbose and content_parts:
            print()  # Newline after streaming content

        # Add newline after tool call streaming
        if self.verbose and tool_calls_dict:
            print()  # Newline after streaming tool calls

        # Build the complete message object
        complete_content = "".join(content_parts) if content_parts else None
        tool_calls_list = [tool_calls_dict[i] for i in sorted(tool_calls_dict.keys())] if tool_calls_dict else None

        # Removed "Accumulated X tool call(s)" print for cleaner output

        # Create a message object that mimics the non-streaming format
        class Message:
            def __init__(self, content, tool_calls):
                self.content = content
                self.tool_calls = []

                if tool_calls:
                    for tc in tool_calls:
                        # Create tool call objects
                        class ToolCall:
                            def __init__(self, id, call_type, function_name, function_args):
                                self.id = id
                                self.type = call_type
                                # Create a simple object to hold function name and arguments
                                class FunctionObj:
                                    def __init__(self, name, arguments):
                                        self.name = name
                                        self.arguments = arguments
                                self.function = FunctionObj(function_name, function_args)

                        self.tool_calls.append(
                            ToolCall(
                                tc["id"],
                                tc["type"],
                                tc["function"]["name"],
                                tc["function"]["arguments"]
                            )
                        )

        return Message(complete_content, tool_calls_list)

    def process_prompt(self, user_prompt: str) -> str:
        """
        Process a user prompt and return the agent's response.

        This method runs the agentic loop, allowing the LLM to:
        1. Understand the user's request
        2. Call tools to inspect data or execute code
        3. Generate and run pandas code
        4. Return results

        Args:
            user_prompt: The user's question or request

        Returns:
            The agent's final response as a string
        """
        if self.verbose:
            print("=" * 70)
            print(f"USER PROMPT: {user_prompt}")
            print("=" * 70)
            print()

        # Add user message to conversation
        self.messages.append({
            "role": "user",
            "content": self.user_prompt + user_prompt
        })

        # Run the agentic loop
        response = self._run_agent_loop()

        return response

    def _run_agent_loop(self) -> str:
        """
        Internal method to run the agentic loop with tool calling.

        Returns:
            The final response from the agent
        """
        for iteration in range(self.max_iterations):
            # Removed iteration print for cleaner output

            # Build messages with dynamic dataframe context
            messages_with_context = self.messages.copy()

            # Inject dataframe context if any dataframes exist
            df_context = get_dataframe_context(self.dataframes)
            if df_context:
                # Add context as a system message after the initial system message
                messages_with_context.insert(1, {
                    "role": "system",
                    "content": df_context
                })

            # Force natural language response when awaiting final response
            if self.awaiting_final_response:
                messages_with_context.append({
                    "role": "user",
                    "content": "Please summarize the results above in natural language. Do not make any tool calls."
                })

            #print(messages_with_context)
            # Call the LLM
            request_kwargs = {
                "model": self.model,
                "messages": messages_with_context,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": self.stream
            }

            if self.force_final_response_after_success and self.awaiting_final_response:
                if self.verbose:
                    print("  âš™ï¸  Requesting final response without tool access")
            else:
                request_kwargs["tools"] = ALL_TOOLS

            response = self.client.chat.completions.create(**request_kwargs)

            # Handle streaming vs non-streaming response
            if self.stream:
                assistant_message = self._process_streaming_response(response)
            else:
                assistant_message = response.choices[0].message

            # Check if there are tool calls
            if assistant_message.tool_calls:
                # Removed "LLM requested X tool call(s)" print for cleaner output

                # Add assistant message to conversation
                # Convert tool_calls to dicts for JSON serialization
                tool_calls_dict = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]

                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": tool_calls_dict
                })

                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name

                    # Parse arguments - handle both string (JSON) and dict formats
                    try:
                        if isinstance(tool_call.function.arguments, str):
                            arguments = json.loads(tool_call.function.arguments)
                        elif isinstance(tool_call.function.arguments, dict):
                            arguments = tool_call.function.arguments
                        else:
                            raise TypeError(f"Unexpected arguments type: {type(tool_call.function.arguments)}")
                    except json.JSONDecodeError as e:
                        # Handle malformed JSON from streaming
                        error_msg = f"Failed to parse tool arguments: {e}"
                        if self.verbose:
                            print(f"\n    ❌ {error_msg}")
                            print(f"    Raw arguments: {repr(tool_call.function.arguments)[:500]}")
                            print(f"    Skipping this tool call and continuing...")

                        # Add error message to conversation so LLM can see what went wrong
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps({
                                "success": False,
                                "error": error_msg,
                                "error_type": "JSONDecodeError",
                                "raw_arguments": str(tool_call.function.arguments)[:500]
                            })
                        })
                        # Continue to next tool call
                        continue

                    # Auto-add print() to last line if needed
                    if tool_name == "execute_python_code":
                        arguments = ensure_print_in_code(arguments)

                    if self.verbose:
                        print(f"  -> Calling tool: {tool_name}")
                        if tool_name in ["execute_python_code"]:
                            # Show execution parameters
                            use_gpu = arguments.get("use_gpu", True)
                            print(f"    use_gpu: {use_gpu}")
                            print(f"    code:")
                            # Display the code with indentation
                            code_lines = arguments.get("code", "").split("\n")
                            for line in code_lines:
                                print(f"      {line}")
                        else:
                            print(f"    Arguments: {json.dumps(arguments, indent=6)}")

                    # Execute the tool
                    # When skip_final_response is True, suppress tool's verbose output
                    # since we'll display it in the highlighted [TOOL OUTPUT] section
                    if tool_name == "execute_python_code" and self.skip_final_response:
                        arguments['verbose'] = False

                    result = call_tool(tool_name, **arguments)

                    # For successful executions, only show minimal info
                    if result.get("success") and tool_name == "execute_python_code":
                        minimal_result = {
                            "success": result.get("success"),
                            "mode": result.get("mode"),
                            "execution_time_seconds": result.get("execution_time_seconds")
                        }
                        print(truncate_output(minimal_result))
                    else:
                        # For failures or other tools, show full output
                        print(truncate_output(result))

                    # Store execution result if it's a code execution tool
                    if tool_name in ["execute_python_code"]:
                        self.last_execution_result = result

                        # Update dataframe tracking from execution result
                        if result.get("success") and "dataframes" in result:
                            self.dataframes = result["dataframes"]

                            # Show dataframe changes if verbose
                            if self.verbose and "dataframe_changes" in result:
                                changes = result["dataframe_changes"]
                                if changes.get("added"):
                                    print(f"    DataFrames created: {', '.join(changes['added'])}")
                                if changes.get("modified"):
                                    print(f"    DataFrames modified: {', '.join(changes['modified'])}")
                                if changes.get("removed"):
                                    print(f"    DataFrames removed: {', '.join(changes['removed'])}")

                        # Exit immediately if successful execution without print output
                        if result.get("success"):
                            if self.force_final_response_after_success:
                                self.awaiting_final_response = True

                            stdout = result.get("stdout", "")
                            output = result.get("output", "")
                            if not stdout and not output:
                                if self.verbose:
                                    print("No output from execution, exiting")
                                self.awaiting_final_response = False
                                return ""

                    if self.verbose and tool_name not in ["execute_python_code"]:
                        print(f"    Result: {truncate_output(result)}")

                    # Add tool result to conversation
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result)
                    })

                    # Clean up failed executions if this execution succeeded
                    if result.get("success", False):
                        original_count = len(self.messages)
                        self.messages = clean_failed_executions(self.messages)
                        removed_count = original_count - len(self.messages)
                        # Removed "Cleaned X failed execution(s)" print for cleaner output

                    # After code execution, decide whether to continue loop or return
                    if tool_name in ["execute_python_code"]:
                        # Only skip final response if BOTH:
                        # 1. skip_final_response flag is True
                        # 2. Execution was successful
                        if self.skip_final_response and result.get("success"):
                            # Removed "Code execution complete" print for cleaner output

                            # Extract stdout and output from the result
                            stdout = result.get("stdout", "")
                            output = result.get("output", "")
                            final_output = stdout if stdout else output

                            # if self.verbose and final_output:
                            #     print("[TOOL OUTPUT]")
                            #     print("-" * 70)
                            #     print(final_output)
                            #     print("-" * 70)
                            #     print()

                            return final_output
                        else:
                            # Continue the loop in these cases:
                            # 1. skip_final_response=False (need LLM summary)
                            # 2. Execution failed (LLM needs to see error and retry)
                            # Removed status prints for cleaner output

                            # Continue the loop to get the LLM's response
                            break
            else:
                # No more tool calls, we have the final response
                final_content = clean_assistant_content(assistant_message.content or "(No response)")
                self.awaiting_final_response = False

                # Check if content looks like a tool call JSON (even though tools were disabled)
                # If so, treat it as empty response
                if final_content.strip().startswith("{") and "tool_calls" in final_content:
                    final_content = ""

                # Add assistant's final message to conversation
                self.messages.append({
                    "role": "assistant",
                    "content": final_content
                })

                # Remove the successful execution that was just processed
                # The LLM has already incorporated the result into its response
                original_count = len(self.messages)
                self.messages = remove_last_successful_execution(self.messages)
                removed_count = original_count - len(self.messages)
                # Removed "Removed processed execution" print for cleaner output

                if self.verbose:
                    print("\n[AGENT RESPONSE]")
                    print("-" * 70)
                    print(final_content)
                    print("-" * 70)
                    print()

                return final_content

        # Max iterations reached
        warning = f" Warning: Reached maximum iterations ({self.max_iterations})"
        if self.verbose:
            print(f"\n{warning}")
        return warning

    def reset_conversation(self):
        """
        Reset the conversation history and execution environment, starting fresh.
        Useful when you want to start a new analysis.
        Clears both the chat history and all variables (including dataframes).
        """
        self.messages = [
            {
                "role": "system",
                "content": self.sys_prompt
            }
        ]
        self.last_execution_result = None
        self.dataframes = {}
        self.awaiting_final_response = False

        # Reset the execution environment (clears all variables)
        reset_result = reset_execution_environment()

        if self.verbose:
            print("Conversation and execution environment reset.")
            if reset_result.get("success"):
                print("  All variables cleared from execution environment.")

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """
        Get the full conversation history.

        Returns:
            List of message dictionaries
        """
        return self.messages.copy()

    def get_last_execution_result(self) -> Optional[Dict[str, Any]]:
        """
        Get the result of the last code execution.

        Returns:
            Dictionary with execution results or None if no code was executed
        """
        return self.last_execution_result

    def get_context_tokens(self) -> Dict[str, Any]:
        """
        Calculate the approximate number of tokens in the current context.

        Returns:
            Dictionary containing token statistics for the conversation history
        """
        return calculate_context_tokens(self.messages)