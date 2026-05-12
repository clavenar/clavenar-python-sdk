"""Configuration surface for `warden_wrap`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from warden_ai.errors import WardenTransportError
    from warden_ai.transport import WardenVerdict


WardenMode = Literal["enforce", "observe"]


@dataclass(frozen=True)
class WardenVerdictContext:
    """Context passed to `WardenOptions.on_verdict` and `on_policy_error`."""

    tool_name: str
    tool_use_id: str
    tool_input: Any


@dataclass
class WardenOptions:
    """Configuration for `warden_wrap`.

    `endpoint` is the warden-lite ingress URL. `token` is the optional
    shared bearer set via `WARDEN_LITE_TOKEN`. `mode` mirrors
    `WARDEN_MODE` on the server: `observe` inspects + logs but never
    blocks, even if warden is unreachable.

    `on_verdict` fires once per inspected tool_use with the verdict
    warden returned, before any deny→raise translation.

    `on_policy_error` fires when an inspection fails at the transport
    layer in `observe` mode (warden unreachable, malformed body, …).
    The agent call passes through as if the tool were allowed,
    preserving the SDK's observe contract even when warden is down.
    Not invoked in `enforce` mode — that path raises the transport
    error (fail-closed).
    """

    endpoint: str
    token: str | None = None
    mode: WardenMode = "enforce"
    timeout_s: float = 10.0
    on_verdict: (
        Callable[[WardenVerdict, WardenVerdictContext], Awaitable[None] | None] | None
    ) = None
    on_policy_error: (
        Callable[
            [WardenTransportError, WardenVerdictContext],
            Awaitable[None] | None,
        ]
        | None
    ) = None
    # Custom HTTP headers forwarded on every inspect call. Empty by
    # default. Useful for proxy auth, tenant routing, demo-prefix
    # tags — anything that needs to ride along with `Authorization`.
    extra_headers: dict[str, str] = field(default_factory=dict)
