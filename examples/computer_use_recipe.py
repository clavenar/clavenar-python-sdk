"""Anthropic Computer Use + warden — gate every action a computer-using
agent takes (mouse click, keystroke, shell command, file edit) before
it reaches the workstation.

Computer Use ships three high-blast-radius tool types — ``computer``,
``bash``, and ``str_replace_editor``. Each lands as a normal Anthropic
``tool_use`` block, so wrapping the client with :func:`warden_wrap` is
the entire integration. What matters is the policy: extend your Rego
with rules keyed off ``input.params.name`` to deny what shouldn't
execute.

Usage:
    pip install warden-ai anthropic
    python examples/computer_use_recipe.py
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from warden_ai import (
    WardenDenied,
    WardenOptions,
    WardenPending,
    warden_wrap,
)


class _StubAnthropicMessages:
    """Stand-in for ``AsyncAnthropic().messages`` so this recipe stays
    dependency-free. Replace with the real client at deployment.
    """

    async def create(self, **_: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(
            id="msg-stub",
            type="message",
            role="assistant",
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="tu-bash-1",
                    name="bash",
                    input={"command": "rm -rf /var/www/staging"},
                )
            ],
            stop_reason="tool_use",
            model="claude-haiku-4-5",
        )


class _StubAnthropic:
    def __init__(self) -> None:
        self.messages = _StubAnthropicMessages()


async def main() -> None:
    options = WardenOptions(
        endpoint=os.environ.get("WARDEN_LITE_URL", "http://localhost:8088"),
        token=os.environ.get("WARDEN_LITE_TOKEN", "demo-token"),
        mode="enforce",
    )

    # Real wiring:
    #   from anthropic import AsyncAnthropic
    #   anthropic = AsyncAnthropic()
    anthropic = _StubAnthropic()
    wrapped = warden_wrap(anthropic, options)

    try:
        msg = await wrapped.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            tools=[
                {
                    "type": "computer_20250124",
                    "name": "computer",
                    "display_width_px": 1024,
                    "display_height_px": 768,
                },
                {"type": "bash_20250124", "name": "bash"},
                {"type": "text_editor_20250124", "name": "str_replace_editor"},
            ],
            messages=[
                {"role": "user", "content": "Clean up the staging directory."}
            ],
        )
        names = ", ".join(getattr(b, "type", "?") for b in msg.content)
        print(f"green — content blocks: {names}")
    except WardenDenied as denied:
        print(f"deny ({denied.tool_name}): {' ; '.join(denied.reasons)}")
    except WardenPending as pending:
        print(f"pending ({pending.correlation_id}) — awaiting operator")
        try:
            await pending.resolve()
            print("resolved: allow")
        except WardenDenied as decided:
            print(f"resolved: deny — {' ; '.join(decided.reasons)}")


if __name__ == "__main__":
    asyncio.run(main())
