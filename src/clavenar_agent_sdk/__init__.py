"""Wrap your Anthropic / OpenAI client with Clavenar inspection.

Supports async (`AsyncAnthropic`, `AsyncOpenAI`) and sync
(`Anthropic`, `OpenAI`) clients, with non-streaming and streaming
responses. Tool calls are inspected by clavenar-lite before the partner
sees them; a denied call raises `ClavenarDenied` (mid-iteration for
streams), a parked call raises `ClavenarPending` with an `await
.resolve()` helper that blocks until an operator decides, and a 429
from the velocity / spend gates raises `ClavenarRateLimited`.
"""

from clavenar_agent_sdk.devmode import render_deny_panel
from clavenar_agent_sdk.errors import (
    ClavenarConfigError,
    ClavenarDenied,
    ClavenarPending,
    ClavenarRateLimited,
    ClavenarRecoveryRequired,
    ClavenarTransportError,
)
from clavenar_agent_sdk.governed_execution import (
    AsyncDurableExecutionStore,
    AsyncGovernedExecutionOptions,
    ExecutionEffect,
    ExecutionState,
    GovernedExecutionOutcome,
    PreparedToolRequest,
    SyncDurableExecutionStore,
    SyncGovernedExecutionOptions,
    ToolExecutionRequest,
    execute_prepared_tool,
    execute_prepared_tool_sync,
    execute_tool,
    execute_tool_sync,
)
from clavenar_agent_sdk.options import (
    ClavenarOptions,
    ClavenarRetryOptions,
    ClavenarVerdictContext,
)
from clavenar_agent_sdk.realtime import (
    inspect_realtime_function_call,
    is_realtime_function_call_done,
    normalize_realtime_function_call,
)
from clavenar_agent_sdk.secure_transport import ProxyPolicy, SecureTransportProfile, TokenProvider
from clavenar_agent_sdk.stream import (
    wrap_anthropic_stream,
    wrap_anthropic_stream_sync,
    wrap_openai_chat_stream,
    wrap_openai_chat_stream_sync,
)
from clavenar_agent_sdk.transport import (
    ClavenarVerdict,
    NormalizedToolCall,
    inspect_tool_use,
    inspect_tool_use_sync,
    inspect_tool_uses,
    inspect_tool_uses_sync,
    poll_pending_once,
    poll_pending_once_sync,
)
from clavenar_agent_sdk.wrap import clavenar_wrap

__version__ = "1.5.3"

__all__ = [
    "AsyncDurableExecutionStore",
    "AsyncGovernedExecutionOptions",
    "ClavenarConfigError",
    "ClavenarDenied",
    "ClavenarOptions",
    "ClavenarPending",
    "ClavenarRateLimited",
    "ClavenarRecoveryRequired",
    "ClavenarRetryOptions",
    "ClavenarTransportError",
    "ClavenarVerdict",
    "ClavenarVerdictContext",
    "ExecutionEffect",
    "ExecutionState",
    "GovernedExecutionOutcome",
    "NormalizedToolCall",
    "PreparedToolRequest",
    "ProxyPolicy",
    "SecureTransportProfile",
    "SyncDurableExecutionStore",
    "SyncGovernedExecutionOptions",
    "TokenProvider",
    "ToolExecutionRequest",
    "__version__",
    "clavenar_wrap",
    "execute_prepared_tool",
    "execute_prepared_tool_sync",
    "execute_tool",
    "execute_tool_sync",
    "inspect_realtime_function_call",
    "inspect_tool_use",
    "inspect_tool_use_sync",
    "inspect_tool_uses",
    "inspect_tool_uses_sync",
    "is_realtime_function_call_done",
    "normalize_realtime_function_call",
    "poll_pending_once",
    "poll_pending_once_sync",
    "render_deny_panel",
    "wrap_anthropic_stream",
    "wrap_anthropic_stream_sync",
    "wrap_openai_chat_stream",
    "wrap_openai_chat_stream_sync",
]
