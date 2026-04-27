"""
Monkey-patch for langchain_anthropic to handle proxy API returning
tool_use.input as JSON string instead of dict.
Must be imported BEFORE any LLM calls.
"""

import json

from langchain_core.messages.tool import tool_call
from langchain_anthropic import output_parsers as _op

def _patched_extract_tool_calls(content):
    """Patched version that parses string args from proxy API."""
    if isinstance(content, list):
        tool_calls_list = []
        for block in content:
            if isinstance(block, str):
                continue
            if block.get("type") != "tool_use":
                continue
            args = block.get("input", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            tool_calls_list.append(
                tool_call(name=block["name"], args=args, id=block["id"]),
            )
        return tool_calls_list
    return []

_op.extract_tool_calls = _patched_extract_tool_calls
