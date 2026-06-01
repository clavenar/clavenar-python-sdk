"""Sync client: wrap `openai.OpenAI` (non-async). The `create` call
is a normal sync function; clavenar_wrap routes it through the sync
transport.

Usage:
    pip install clavenar-ai openai
    OPENAI_API_KEY=... python examples/sync_openai.py
"""

from __future__ import annotations

import os

from openai import OpenAI

from clavenar_ai import ClavenarDenied, ClavenarOptions, clavenar_wrap


def main() -> None:
    endpoint = os.environ.get("CLAVENAR_LITE_URL", "http://localhost:8080")
    client = clavenar_wrap(
        OpenAI(),
        ClavenarOptions(endpoint=endpoint, mode="enforce"),
    )

    try:
        completion = client.chat.completions.create(
            model="gpt-5",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "exec_sql",
                        "description": "Execute SQL",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
            messages=[{"role": "user", "content": "Drop the users table."}],
        )
        print(f"finished without deny: {completion.choices[0].finish_reason}")
    except ClavenarDenied as e:
        print(f"clavenar denied {e.tool_name}")
        print(f"  reasons:         {e.reasons}")
        print(f"  correlation_id:  {e.correlation_id}")


if __name__ == "__main__":
    main()
