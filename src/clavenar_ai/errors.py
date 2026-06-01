"""Exception hierarchy mirroring `@vanteguardlabs/clavenar-ai-sdk` 0.3.0.

A partner catching `ClavenarDenied` / `ClavenarPending` in Python should
see the same fields they'd see in the TS SDK — name, reasons, review
reasons, intent category, correlation id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from clavenar_ai.transport import ClavenarPendingView


class ClavenarConfigError(Exception):
    """Malformed config — bad endpoint URL, wrong client kind, etc."""


class ClavenarTransportError(Exception):
    """Clavenar is unreachable, returned an unexpected status, or sent a malformed body."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ClavenarDenied(Exception):
    """Raised when clavenar returns a 403 security_violation."""

    def __init__(
        self,
        *,
        tool_name: str,
        reasons: list[str],
        review_reasons: list[str],
        intent_category: str,
        correlation_id: str | None = None,
    ) -> None:
        super().__init__(f"clavenar denied tool {tool_name!r}: {' | '.join(reasons)}")
        self.tool_name = tool_name
        self.reasons = reasons
        self.review_reasons = review_reasons
        self.intent_category = intent_category
        self.correlation_id = correlation_id


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

        Transient transport errors (5xx, network blips) are swallowed
        between polls. Terminal failures (401, 404, body-shape
        mismatch) re-raise immediately as ClavenarTransportError. The
        deadline is enforced as a hard wall-clock ceiling.
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
