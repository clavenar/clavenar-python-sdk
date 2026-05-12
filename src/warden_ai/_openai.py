"""OpenAI-specific extraction of tool calls from a ChatCompletion.

Same dual-access trick as `_anthropic.py` — accepts both dicts and
Pydantic-model-shaped responses without importing `openai` at runtime.
"""

from __future__ import annotations

import json
from typing import Any

from warden_ai.transport import NormalizedToolCall


def extract_tool_calls(result: Any) -> list[NormalizedToolCall]:
    choices = _get(result, "choices", default=[])
    if not isinstance(choices, list):
        return []
    out: list[NormalizedToolCall] = []
    for choice in choices:
        message = _get(choice, "message")
        if message is None:
            continue
        tool_calls = _get(message, "tool_calls", default=[])
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            normalized = _normalize_chat_tool_call(call)
            if normalized is not None:
                out.append(normalized)
    return out


def _normalize_chat_tool_call(call: Any) -> NormalizedToolCall | None:
    if _get(call, "type") != "function":
        return None
    call_id = _get(call, "id")
    if not isinstance(call_id, str):
        return None
    function = _get(call, "function")
    if function is None:
        return None
    name = _get(function, "name")
    arguments_raw = _get(function, "arguments")
    if not isinstance(name, str) or not isinstance(arguments_raw, str):
        return None
    try:
        arguments = json.loads(arguments_raw) if arguments_raw else {}
    except json.JSONDecodeError:
        # Pass through as a string — warden-lite's MCP envelope
        # accepts arbitrary JSON values; preserving the literal lets
        # the inspector see what the model actually emitted, even if
        # malformed.
        arguments = arguments_raw
    return NormalizedToolCall(id=call_id, name=name, input=arguments)


def _get(obj: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
