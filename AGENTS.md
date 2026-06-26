<!-- public repo — do not add internal topology, secrets, deploy/runbook, strategy, or absolute host paths -->
# clavenar-python-sdk — agent-side wrapper SDK (PyPI `clavenar-agent-sdk`): async + sync mirror of the TS wrapper

Wrap an Anthropic / OpenAI client (async or sync). Every tool call the
model emits is inspected by Clavenar **before** the agent loop can run
it — a denied call raises instead of executing. Pure client: no server,
no listener; it talks to a Clavenar gateway over HTTP.

## Build, test, lint
- Dev install: `pip install -e '.[dev]'`
- Build dist: `python -m build` (sdist + wheel; backend `hatchling`)
- Test: `pytest` (`asyncio_mode = "auto"`, `testpaths = ["tests"]`)
- Lint: `ruff check` + `ruff format --check`
- Types: `mypy src/clavenar_agent_sdk` (`strict`, pinned to py3.10 floor)

Python 3.10+. Runtime dep is `httpx` only (plus `typing-extensions` on
<3.12). `anthropic` / `openai` are **not** imported — bring your own.

Run: library, no binary. Public entry: `clavenar_wrap(client, ClavenarOptions(...))`.
The wrapped client targets a Clavenar gateway at `ClavenarOptions.endpoint`
(e.g. clavenar-lite on `http://localhost:8080`); no port is opened by the SDK.

## Layout
- `src/clavenar_agent_sdk/__init__.py` — public surface; `__all__` is the API contract. Mirror any export change here.
- `wrap.py` — `clavenar_wrap`: in-place monkey-patch of the client's `create`; sync/async fork via `inspect.iscoroutinefunction`.
- `transport.py` — inspect calls + pending poll (`inspect_tool_use[_sync]`, `poll_pending_once[_sync]`); `NormalizedToolCall`, `ClavenarVerdict`.
- `stream.py` — streaming intercept; closing event held until verdict (`wrap_anthropic_stream[_sync]`, `wrap_openai_chat_stream[_sync]`).
- `_anthropic.py` / `_openai.py` — provider-shaped tool-call extraction (structural, no provider import).
- `realtime.py` — standalone OpenAI Realtime WS helpers (`inspect_realtime_function_call`, …).
- `options.py` — `ClavenarOptions`, `ClavenarRetryOptions`, `ClavenarVerdictContext`.
- `errors.py` — `ClavenarDenied` / `ClavenarPending` / `ClavenarTransportError` / `ClavenarConfigError`.
- `devmode.py` — `render_deny_panel` (stderr deny panel when `dev_mode=True`).
- `tests/`, `examples/` (anthropic/openai/langchain/llamaindex/realtime recipes), `docs/SEQUENCES.md`.

## Conventions & invariants
- **Inspect-before-run is the whole product.** Every tool call is inspected before the partner code can act on it; never add a path that runs a tool ahead of its verdict.
- **No provider import.** Detect client shape structurally (duck-typing on `create`); keep `anthropic`/`openai` optional — don't import them at module load.
- **Streaming gate.** Events pass through in order, but the closing event (Anthropic `content_block_stop`, OpenAI `finish_reason="tool_calls"`) is withheld until the verdict; a deny raises mid-iteration.
- **`.stream()` helpers are blocked.** Provider `messages.stream()` / `chat.completions.stream()` can't be wrapped faithfully → `ClavenarConfigError` unless caller sets `allow_uninspected_stream=True`. Don't quietly let tool calls bypass inspection.
- **Modes.** `enforce` (default) raises on deny / on transport failure after retries; `observe` passes through and fires `on_verdict` / `on_policy_error`.
- **Pending → resolve.** `ClavenarPending.resolve()` polls; transient 5xx / network blips are swallowed between polls, terminal 4xx (404/401) re-raise.
- **Retries.** 5xx + network errors retry with jittered exponential backoff; 200/403/other-4xx never retry. `max_attempts=1` disables.
- **`dev_mode` is an attacker oracle.** Per-detector deny detail is gated; only enable verbose verdicts / `dev_mode` in dev/staging.
- **Sync callbacks for sync clients.** `on_verdict` / `on_policy_error` must be sync when wrapping a sync client.
- Keep TS/Python parity: this SDK is a faithful client of the same wire contract the TS sibling implements; fix wire divergences against the spec, not by forking behaviour here.

Python coding standards (the ones that bite here):
- `ruff check` + `ruff format --check` clean; line length 100; lint set `E,F,I,W,B,UP,SIM,RUF`.
- `mypy --strict` clean; the package ships `py.typed` — keep the public API fully typed.
- Tests live in `tests/`; `asyncio_mode = "auto"` so `async def test_*` needs no marker. Use `respx` to stub `httpx`.
- Comments: write none by default; one short line only when the *why* is non-obvious. Don't reference tasks/PRs in source.
- Fix root causes, not symptoms — never silence a lint or type error to make it pass.

## Pointers
README.md · SECURITY.md · docs/SEQUENCES.md · examples/ — the wire contract this SDK speaks (POST /mcp, verdict envelope, pending/resolve, `X-Clavenar-*` headers) is the source of truth in the public `clavenar-specs` TECH_SPEC.
