"""Minimal example: wrap AsyncAnthropic with warden-ai and catch a deny.

Run with a real Anthropic key + a real warden-lite at the endpoint.
The /demo curated catalog ships a `sql_execute` scenario that
warden denies in policy — this script catches the deny and prints
the reasons + correlation id.

Usage:
    pip install warden-ai anthropic
    ANTHROPIC_API_KEY=... python examples/basic_anthropic.py
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
        result = await client.messages.create(
            model="claude-opus-4-7",
            max_tokens=512,
            tools=[
                {
                    "name": "sql_execute",
                    "description": "Execute SQL against the production DB",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": "Drop the users table.",
                }
            ],
        )
        print(f"agent finished without deny: {result.stop_reason}")
    except WardenDenied as e:
        print(f"warden denied {e.tool_name}")
        print(f"  reasons:          {e.reasons}")
        print(f"  intent_category:  {e.intent_category}")
        print(f"  correlation_id:   {e.correlation_id}")


if __name__ == "__main__":
    asyncio.run(main())
