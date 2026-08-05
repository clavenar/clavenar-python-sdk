"""OpenAI-specific extraction of tool calls from a ChatCompletion.

Same dual-access trick as `_anthropic.py` — accepts both dicts and
Pydantic-model-shaped responses without importing `openai` at runtime.
"""

from __future__ import annotations

import json
from typing import Any

from clavenar_agent_sdk.errors import ClavenarTransportError
from clavenar_agent_sdk.transport import NormalizedToolCall


def extract_tool_calls(result: Any) -> list[NormalizedToolCall]:
    choices = _get(result, "choices")
    if not isinstance(choices, list):
        raise ClavenarTransportError("OpenAI response is missing its choices list")
    out: list[NormalizedToolCall] = []
    for choice in choices:
        before = len(out)
        message = _get(choice, "message")
        if message is None:
            raise ClavenarTransportError("OpenAI response contains a choice without a message")
        tool_calls = _get(message, "tool_calls", default=[])
        if not isinstance(tool_calls, list):
            if _get(choice, "finish_reason") == "tool_calls":
                raise ClavenarTransportError(
                    "OpenAI response declared tool_calls but tool_calls is not a list"
                )
            continue
        for call in tool_calls:
            normalized = _normalize_chat_tool_call(call)
            out.append(normalized)
        if _get(choice, "finish_reason") == "tool_calls" and len(out) == before:
            raise ClavenarTransportError(
                "OpenAI response declared tool_calls but contained no valid function call"
            )
    return out


def _normalize_chat_tool_call(call: Any) -> NormalizedToolCall:
    if _get(call, "type") != "function":
        raise ClavenarTransportError("OpenAI tool_call has an unsupported or missing type")
    call_id = _get(call, "id")
    if not isinstance(call_id, str) or not call_id:
        raise ClavenarTransportError("OpenAI tool_call is missing a valid id")
    function = _get(call, "function")
    if function is None:
        raise ClavenarTransportError("OpenAI tool_call is missing function metadata")
    name = _get(function, "name")
    arguments_raw = _get(function, "arguments")
    if not isinstance(name, str) or not name or not isinstance(arguments_raw, str):
        raise ClavenarTransportError(
            "OpenAI tool_call function is missing a valid name or arguments string"
        )
    try:
        arguments = json.loads(arguments_raw) if arguments_raw else {}
    except json.JSONDecodeError as error:
        raise ClavenarTransportError(
            f"OpenAI tool_call {call_id} emitted malformed JSON arguments: {error}"
        ) from error
    return NormalizedToolCall(id=call_id, name=name, input=arguments)


def _get(obj: Any, key: str, *, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
