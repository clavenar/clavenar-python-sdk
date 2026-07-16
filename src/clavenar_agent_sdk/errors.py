"""Exception hierarchy mirroring `@clavenar/agent-sdk` 1.1.0.

A partner catching `ClavenarDenied` / `ClavenarPending` in Python should
see the same fields they'd see in the TS SDK — name, reasons, review
reasons, intent category, correlation id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from clavenar_agent_sdk.transport import ClavenarPendingView


class ClavenarConfigError(Exception):
    """Malformed config — bad endpoint URL, wrong client kind, etc."""


class ClavenarTransportError(Exception):
    """Clavenar is unreachable, returned an unexpected status, or sent a malformed body."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ClavenarDenied(Exception):
    """Raised when clavenar returns a 403 deny."""

    def __init__(
        self,
        *,
        tool_name: str,
        reasons: list[str],
        review_reasons: list[str],
        intent_category: str,
        layer: str | None = None,
        correlation_id: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        super().__init__(f"clavenar denied tool {tool_name!r}: {' | '.join(reasons)}")
        self.tool_name = tool_name
        self.reasons = reasons
        self.review_reasons = review_reasons
        self.intent_category = intent_category
        # Stage that produced the deny (brain / policy / hil / egress /
        # …) when the server reports it; None for older servers or
        # operator-driven pending denials.
        self.layer = layer
        self.correlation_id = correlation_id
        # Per-detector verbose-verdict breakdown when the gateway opts in
        # (CLAVENAR_PROXY_VERBOSE_VERDICTS=true); None otherwise. Shape:
        # {"detectors": [{"detector", "score", "flagged"?}], "degraded": [..]}.
        self.detail = detail


class ClavenarPending(Exception):
    """Raised when clavenar parks a tool call for human review (202 yellow tier).

    Catch and `await pending.resolve()` to block until an operator
    decides. `resolve()` returns cleanly on allow and re-raises
    `ClavenarDenied` on deny — same control flow as the synchronous
    path, so a try/except wrapping the agent call covers both.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        correlation_id: str,
        review_reasons: list[str],
        poll_once: Callable[[], Awaitable[ClavenarPendingView]],
    ) -> None:
        super().__init__(
            f"clavenar parked tool {tool_name!r} for review (correlation_id={correlation_id})"
        )
        self.tool_name = tool_name
        self.correlation_id = correlation_id
        self.review_reasons = review_reasons
        self._poll_once = poll_once

    async def resolve(
        self,
        *,
        poll_interval_s: float = 2.0,
        timeout_s: float = 600.0,
    ) -> None:
        """Block until an operator decides. Returns on allow; raises ClavenarDenied on deny.

        Transient transport errors (5xx, network blips, and body-shape
        mismatches — the latter carry the poll's 200 status) are
        swallowed between polls. Only 401/404 are terminal and re-raise
        immediately as ClavenarTransportError. The deadline is enforced
        as a hard wall-clock ceiling.
        """
        import asyncio
        import time

        if poll_interval_s <= 0:
            raise ClavenarTransportError(
                f"ClavenarPending.resolve: poll_interval_s must be positive, got {poll_interval_s}"
            )
        if timeout_s <= 0:
            raise ClavenarTransportError(
                f"ClavenarPending.resolve: timeout_s must be positive, got {timeout_s}"
            )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            view: ClavenarPendingView | None = None
            try:
                view = await self._poll_once()
            except ClavenarTransportError as e:
                if e.status in (401, 404):
                    raise

            if view is not None and view.decision == "allow":
                return
            if view is not None and view.decision == "deny":
                reasons = [view.decider_note] if view.decider_note else ["operator denied"]
                raise ClavenarDenied(
                    tool_name=self.tool_name,
                    reasons=reasons,
                    review_reasons=self.review_reasons,
                    intent_category="PendingDenied",
                    correlation_id=self.correlation_id,
                )

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            await asyncio.sleep(min(poll_interval_s, remaining))

        raise ClavenarTransportError(
            f"clavenar pending {self.correlation_id} not decided within {timeout_s}s"
        )


class ClavenarRateLimited(Exception):
    """Raised when clavenar returns a 429 — the request was rejected
    *before* evaluation, by the request-velocity gate (`rate_limited`)
    or the per-tenant spend gate (`quota_exceeded`). Not retried by the
    transport: honor `retry_after_secs` (set on `rate_limited` only) or
    fail the operation.
    """

    def __init__(
        self,
        *,
        tool_name: str,
        code: Literal["rate_limited", "quota_exceeded"],
        reasons: list[str],
        retry_after_secs: int | None = None,
        layer: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        suffix = f" (retry after {retry_after_secs}s)" if retry_after_secs is not None else ""
        super().__init__(f"clavenar {code} for tool {tool_name!r}{suffix}")
        self.tool_name = tool_name
        self.code = code
        self.reasons = reasons
        # Seconds to wait before retrying; None on quota_exceeded.
        self.retry_after_secs = retry_after_secs
        self.layer = layer
        self.correlation_id = correlation_id
