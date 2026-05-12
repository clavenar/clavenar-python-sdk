"""Streaming inspection: warden holds the closing event until a
verdict lands. A denied tool raises mid-iteration before the partner
can act on it.

Usage:
    pip install warden-ai anthropic
    ANTHROPIC_API_KEY=... python examples/streaming_anthropic.py
"""

from __future__ import annotations

import asyncio
import os

from anthropic import AsyncAnthropic

from warden_ai import WardenDenied, WardenOptions, warden_wrap


async def main() -> None:
    endpoint = os.environ.get("WARDEN_LITE_URL", "http://localhost:8080")
    client = warden_wrap(
        AsyncAnthropic(),
        WardenOptions(endpoint=endpoint, mode="enforce"),
    )

    try:
        stream = await client.messages.create(
            model="claude-opus-4-7",
            max_tokens=512,
            stream=True,
            tools=[
                {
                    "name": "sql_execute",
                    "description": "Execute SQL against production",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
            messages=[{"role": "user", "content": "Drop the users table."}],
        )
        async for event in stream:
            print(f"event: {event.type}")
    except WardenDenied as e:
        print(f"warden denied {e.tool_name} mid-stream")
        print(f"  reasons:         {e.reasons}")
        print(f"  correlation_id:  {e.correlation_id}")


if __name__ == "__main__":
    asyncio.run(main())
