"""Shared fixtures: a respx mock router pointing at a fake warden-lite
URL, and lightweight fake Anthropic / OpenAI clients that don't depend
on the real SDKs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from warden_ai.options import WardenOptions

FAKE_ENDPOINT = "http://warden-lite.test"


@pytest.fixture
def opts() -> WardenOptions:
    return WardenOptions(endpoint=FAKE_ENDPOINT, mode="enforce", timeout_s=2.0)


@pytest.fixture
def opts_observe() -> WardenOptions:
    return WardenOptions(endpoint=FAKE_ENDPOINT, mode="observe", timeout_s=2.0)


# ---------------------------------------------------------------------------
# Fake clients. The real `anthropic` and `openai` Python packages aren't
# runtime deps of this SDK; the tests stand up minimal duck-typed clients
# to exercise `warden_wrap`'s detection + interception paths.
# ---------------------------------------------------------------------------


@dataclass
class FakeAnthropicMessages:
    response: Any

    async def create(self, **kwargs: Any) -> Any:
        return self.response


@dataclass
class FakeAnthropicClient:
    messages: FakeAnthropicMessages


@dataclass
class FakeOpenAICompletions:
    response: Any

    async def create(self, **kwargs: Any) -> Any:
        return self.response


@dataclass
class FakeOpenAIChat:
    completions: FakeOpenAICompletions


@dataclass
class FakeOpenAIClient:
    chat: FakeOpenAIChat


def make_anthropic_message_with_tool_use(
    *, tool_id: str = "toolu_001", tool_name: str = "list_files", tool_input: Any | None = None
) -> dict[str, Any]:
    return {
        "id": "msg_001",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "ok, listing"},
            {
                "type": "tool_use",
                "id": tool_id,
                "name": tool_name,
                "input": tool_input or {"path": "/"},
            },
        ],
        "stop_reason": "tool_use",
    }


def make_openai_completion_with_tool_call(
    *, call_id: str = "call_001", name: str = "list_files", arguments: str = '{"path":"/"}'
) -> dict[str, Any]:
    return {
        "id": "chatcmpl-001",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
