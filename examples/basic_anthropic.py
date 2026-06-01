"""Minimal example: wrap AsyncAnthropic with clavenar-ai and catch a deny.

Run with a real Anthropic key + a real clavenar-lite at the endpoint.
The /demo curated catalog ships a `sql_execute` scenario that
clavenar denies in policy — this script catches the deny and prints
the reasons + correlation id.

Usage:
    pip install clavenar-ai anthropic
    ANTHROPIC_API_KEY=... python examples/basic_anthropic.py
"""

from __future__ import annotations

import asyncio
import os

from anthropic import AsyncAnthropic

from clavenar_ai import ClavenarDenied, ClavenarOptions, clavenar_wrap


async def main() -> None:
    endpoint = os.environ.get("CLAVENAR_LITE_URL", "http://localhost:8080")
    client = clavenar_wrap(
        AsyncAnthropic(),
        ClavenarOptions(endpoint=endpoint, mode="enforce"),
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
    except ClavenarDenied as e:
        print(f"clavenar denied {e.tool_name}")
        print(f"  reasons:          {e.reasons}")
        print(f"  intent_category:  {e.intent_category}")
        print(f"  correlation_id:   {e.correlation_id}")


if __name__ == "__main__":
    asyncio.run(main())
