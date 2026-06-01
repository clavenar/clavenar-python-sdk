"""Configuration surface for `clavenar_wrap`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from clavenar_ai.errors import ClavenarTransportError
    from clavenar_ai.transport import ClavenarVerdict


ClavenarMode = Literal["enforce", "observe"]


@dataclass(frozen=True)
class ClavenarRetryOptions:
    """Transient-failure retry policy for `inspect_tool_use`.

    Network errors and 5xx responses retry up to `max_attempts` with
    jittered exponential backoff (`base_delay_s * 2^attempt` ceiling,
    sampled in `[ceiling/2, ceiling]`). 200, 403, and other 4xx never
    retry — they're verdicts or config errors, not transients.

    Set `max_attempts=1` to disable retries entirely. Defaults mirror
    the TS SDK at 0.3.0.
    """

    max_attempts: int = 3
    base_delay_s: float = 0.1


@dataclass(frozen=True)
class ClavenarVerdictContext:
    """Context passed to `ClavenarOptions.on_verdict` and `on_policy_error`."""

    tool_name: str
    tool_use_id: str
    tool_input: Any


@dataclass
class ClavenarOptions:
    """Configuration for `clavenar_wrap`.

    `endpoint` is the clavenar-lite ingress URL. `token` is the optional
    shared bearer set via `CLAVENAR_LITE_TOKEN`. `mode` mirrors
    `CLAVENAR_MODE` on the server: `observe` inspects + logs but never
    blocks, even if clavenar is unreachable.

    `on_verdict` fires once per inspected tool_use with the verdict
    clavenar returned, before any deny→raise translation.

    `on_policy_error` fires when an inspection fails at the transport
    layer in `observe` mode (clavenar unreachable, malformed body, …).
    The agent call passes through as if the tool were allowed,
    preserving the SDK's observe contract even when clavenar is down.
    Not invoked in `enforce` mode — that path raises the transport
    error (fail-closed).
    """

    endpoint: str
    token: str | None = None
    mode: ClavenarMode = "enforce"
    timeout_s: float = 10.0
    on_verdict: Callable[[ClavenarVerdict, ClavenarVerdictContext], Awaitable[None] | None] | None = (
        None
    )
    on_policy_error: (
        Callable[
            [ClavenarTransportError, ClavenarVerdictContext],
            Awaitable[None] | None,
        ]
        | None
    ) = None
    # Custom HTTP headers forwarded on every inspect call. Empty by
    # default. Useful for proxy auth, tenant routing, demo-prefix
    # tags — anything that needs to ride along with `Authorization`.
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Transient-failure retry policy. Defaults to 3 attempts with
    # jittered exponential backoff starting at 100ms.
    retry: ClavenarRetryOptions = field(default_factory=ClavenarRetryOptions)
