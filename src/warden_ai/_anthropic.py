"""Anthropic-specific extraction of tool_use blocks from a Message.

Type-checked structurally to avoid hard-coding the `anthropic` SDK as
a runtime dep. A partner can ship without `anthropic` installed if they
only use the OpenAI side.
"""

from __future__ import annotations

from typing import Any

from warden_ai.transport import NormalizedToolCall


def extract_tool_uses(result: Any) -> list[NormalizedToolCall]:
    """Walk `result.content[]` for `tool_use` blocks. Returns [] if the
    response has no tool_use (text-only completion).
    """
    content = _get(result, "content", default=[])
    if not isinstance(content, list):
        return []
    out: list[NormalizedToolCall] = []
    for block in content:
        if _block_type(block) != "tool_use":
            continue
        block_id = _get(block, "id")
        block_name = _get(block, "name")
        block_input = _get(block, "input")
        if not isinstance(block_id, str) or not isinstance(block_name, str):
            continue
        out.append(
            NormalizedToolCall(id=block_id, name=block_name, input=block_input)
        )
    return out


def _block_type(block: Any) -> str | None:
    t = _get(block, "type")
    return t if isinstance(t, str) else None


def _get(obj: Any, key: str, *, default: Any = None) -> Any:
    """Dual access: dict-key or attribute. The Anthropic SDK exposes
    Pydantic models (attribute access); raw HTTP responses might come
    through as dicts. Both work.
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
