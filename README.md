# warden-ai (Python)

Wrap your async [Anthropic][anthropic] / [OpenAI][openai] Python client.
Every tool call the model emits is inspected by [Agent Warden][warden]
before the agent loop can run it.

```python
import asyncio
from anthropic import AsyncAnthropic

from warden_ai import warden_wrap, WardenDenied, WardenOptions

async def main() -> None:
    client = warden_wrap(
        AsyncAnthropic(),
        WardenOptions(endpoint="http://localhost:8080", mode="enforce"),
    )

    try:
        result = await client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=[...],
            messages=[{"role": "user", "content": "list my files"}],
        )
    except WardenDenied as e:
        print(f"warden denied {e.tool_name}: {e.reasons}")

asyncio.run(main())
```

OpenAI works the same way:

```python
from openai import AsyncOpenAI
from warden_ai import warden_wrap, WardenOptions

client = warden_wrap(
    AsyncOpenAI(),
    WardenOptions(endpoint="http://localhost:8080"),
)

completion = await client.chat.completions.create(
    model="gpt-5",
    tools=[...],
    messages=[...],
)
```

## What this MVP does

- **Inspects every tool call** the model emits, in parallel, before the
  agent loop sees the response. Anthropic `tool_use` blocks and
  OpenAI `tool_calls` entries are both handled.
- **Three verdicts**: allow (pass through), deny (raise
  `WardenDenied`), pending (raise `WardenPending` — catch and
  `await pending.resolve()` to block until an operator decides).
- **Two modes**: `enforce` (default — deny raises; transport failure
  raises) and `observe` (deny passes through; transport failure
  surfaces via `on_policy_error`).
- **Correlation ids** plumbed through so partners can look any
  decision up in the audit ledger.

## What this MVP does NOT do yet

- **Streaming responses** (`stream=True`) pass through unchecked. A
  one-time `RuntimeWarning` fires on first use. Streaming inspection
  arrives in the feature-complete release.
- **Retries on transport failure.** A single attempt; failures
  raise immediately (or route to `on_policy_error` in observe).
- **Sync clients** (`anthropic.Anthropic`, `openai.OpenAI`) — only
  the async variants are wrapped today.

These all land in the Phase 2 sprint-2 feature-complete release,
reaching 1:1 parity with the TS SDK at 0.3.0.

## Install

```bash
pip install warden-ai
```

Python 3.10+. Runtime dep is `httpx` only; the `anthropic` and
`openai` packages are NOT imported by `warden-ai` — bring your own.

## Configuration

| Field | Type | Default | Notes |
|---|---|---|---|
| `endpoint` | `str` | — | warden-lite base URL, e.g. `http://localhost:8080` |
| `token` | `str \| None` | `None` | Shared bearer (`WARDEN_LITE_TOKEN`) |
| `mode` | `"enforce" \| "observe"` | `"enforce"` | Mirror of server-side `WARDEN_MODE` |
| `timeout_s` | `float` | `10.0` | Per-request timeout |
| `on_verdict` | callable \| `None` | `None` | Fired per inspected tool call |
| `on_policy_error` | callable \| `None` | `None` | Fired per transport failure in observe mode |
| `extra_headers` | `dict[str, str]` | `{}` | Forwarded on every inspect (`X-Warden-Demo-Prefix`, proxy auth, …) |

## License

Apache-2.0.

[anthropic]: https://github.com/anthropics/anthropic-sdk-python
[openai]: https://github.com/openai/openai-python
[warden]: https://warden.vanteguardlabs.com
