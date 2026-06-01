# clavenar-ai (Python)

Wrap your [Anthropic][anthropic] / [OpenAI][openai] Python client —
async or sync. Every tool call the model emits is inspected by
[Clavenar][clavenar] before the agent loop can run it.

Sequence diagrams for the five primary paths — `clavenar_wrap` boot
with sync/async fork, async non-streaming inspection, async
streaming choice-end gating, `ClavenarPending.resolve` poll loop, and
the standalone OpenAI Realtime helper — plus a request
decision-tree flowchart, live in
[`docs/SEQUENCES.md`](docs/SEQUENCES.md).

```python
import asyncio
from anthropic import AsyncAnthropic

from clavenar_ai import clavenar_wrap, ClavenarDenied, ClavenarOptions

async def main() -> None:
    client = clavenar_wrap(
        AsyncAnthropic(),
        ClavenarOptions(endpoint="http://localhost:8080", mode="enforce"),
    )

    try:
        result = await client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            tools=[...],
            messages=[{"role": "user", "content": "list my files"}],
        )
    except ClavenarDenied as e:
        print(f"clavenar denied {e.tool_name}: {e.reasons}")

asyncio.run(main())
```

OpenAI works the same way:

```python
from openai import AsyncOpenAI
from clavenar_ai import clavenar_wrap, ClavenarOptions

client = clavenar_wrap(
    AsyncOpenAI(),
    ClavenarOptions(endpoint="http://localhost:8080"),
)

completion = await client.chat.completions.create(
    model="gpt-5",
    tools=[...],
    messages=[...],
)
```

## Sync clients

`anthropic.Anthropic` and `openai.OpenAI` (non-async) are wrapped
exactly the same way — `clavenar_wrap` detects sync vs. async by
inspecting the underlying `create` method:

```python
from anthropic import Anthropic
from clavenar_ai import clavenar_wrap, ClavenarOptions

client = clavenar_wrap(
    Anthropic(),
    ClavenarOptions(endpoint="http://localhost:8080"),
)

result = client.messages.create(model="claude-opus-4-7", ...)
```

Sync clients use `httpx.Client` under the hood and inspections run
serially. Callbacks (`on_verdict`, `on_policy_error`) must be sync
when wrapping a sync client.

## Streaming

`stream=True` is intercepted: each event/chunk passes through in
order, but the closing event (Anthropic `content_block_stop`, OpenAI
`finish_reason="tool_calls"`) is held until clavenar returns a verdict.
A denied tool raises mid-iteration *before* partner code can act on
it.

```python
async with client.messages.create(stream=True, ...) as stream:
    async for event in stream:
        ...
# ClavenarDenied raised inside the async-for if a tool_use was blocked.
```

Both `AsyncAnthropic` + `AsyncOpenAI` streams and their sync
counterparts (`Anthropic`, `OpenAI`) are supported.

## Pending → resolve

When clavenar parks a tool call for human review, `ClavenarPending` is
raised. Catch it and await `resolve()` to block until an operator
decides:

```python
try:
    result = await client.messages.create(...)
except ClavenarPending as p:
    print(f"awaiting approval: {p.review_reasons}")
    await p.resolve(poll_interval_s=2.0, timeout_s=600.0)
    # Returns on approve; raises ClavenarDenied on deny.
```

Transient transport errors (5xx, network blips) are swallowed between
polls. Terminal errors (404, 401) re-raise immediately.

## Retries

Network errors and 5xx responses retry up to `max_attempts` with
jittered exponential backoff. 200, 403, and other 4xx never retry.
Defaults mirror the TS SDK at 0.3.0:

```python
from clavenar_ai import ClavenarOptions, ClavenarRetryOptions

opts = ClavenarOptions(
    endpoint="...",
    retry=ClavenarRetryOptions(max_attempts=3, base_delay_s=0.1),
)
```

Set `max_attempts=1` to disable retries.

## Modes

| Mode | Deny | Transport failure |
|---|---|---|
| `enforce` (default) | raises `ClavenarDenied` | raises `ClavenarTransportError` after retries |
| `observe` | passes through; `on_verdict` fires | passes through; `on_policy_error` fires |

Observe is the rollout knob — surface what clavenar would decide
without breaking the agent. Flip to enforce per-call once verdicts
are trusted.

## Install

```bash
pip install clavenar-ai
```

Python 3.10+. Runtime dep is `httpx` only; the `anthropic` and
`openai` packages are NOT imported by `clavenar-ai` — bring your own.

## Configuration

| Field | Type | Default | Notes |
|---|---|---|---|
| `endpoint` | `str` | — | clavenar-lite base URL, e.g. `http://localhost:8080` |
| `token` | `str \| None` | `None` | Shared bearer (`CLAVENAR_LITE_TOKEN`) |
| `mode` | `"enforce" \| "observe"` | `"enforce"` | Mirror of server-side `CLAVENAR_MODE` |
| `timeout_s` | `float` | `10.0` | Per-request timeout |
| `on_verdict` | callable \| `None` | `None` | Fired per inspected tool call |
| `on_policy_error` | callable \| `None` | `None` | Fired per transport failure in observe mode |
| `extra_headers` | `dict[str, str]` | `{}` | Forwarded on every inspect (`X-Clavenar-Demo-Prefix`, proxy auth, …) |
| `retry` | `ClavenarRetryOptions` | `(3, 0.1)` | Jittered exponential backoff for 5xx + network errors |

## Wire contract

The HTTP shape this SDK speaks against the inspect endpoint
(`POST /inspect`, the verdict envelope, the pending / resolve
contract, and the `X-Clavenar-*` header set) is documented in the
workspace's source of truth:
[`clavenar-specs/TECH_SPEC.md`](https://github.com/clavenar/clavenar-specs/blob/main/TECH_SPEC.md).
This SDK is a faithful client of that contract — if you observe a
divergence, file the bug against the spec first.

The TypeScript sibling at
[`clavenar-ai-sdk`](https://github.com/clavenar/clavenar-ai-sdk)
implements the same wire contract with parity guarantees.

## License

Apache-2.0.

[anthropic]: https://github.com/anthropics/anthropic-sdk-python
[openai]: https://github.com/openai/openai-python
[clavenar]: https://warden.vanteguardlabs.com
