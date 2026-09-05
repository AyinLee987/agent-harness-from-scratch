"""Trigger Layer — decides WHEN and HOW the agent acts.

Components:
    - Gateway: rate limiting, concurrency control, request queuing
    - StateGraph: generic directed graph execution engine
    - ReActLoop: the think → act → observe cycle
    - Dispatch: tool execution with retry and error handling
"""

from .gateway import (
    Admission,
    AgentGateway,
    ConcurrencyGuard,
    RateLimiter,
    RequestQueue,
)
from .graph import StateGraph
from .react_loop import (
    FORCED_REFLECTION_PROMPT,
    REFLECT_AFTER_FAILURE_STATE_KEY,
    ReActLoop,
)
from .dispatch import ToolDispatcher, is_failure_observation
from .router import (
    DIRECT_SYSTEM_PROMPT,
    ESCALATION_SENTINEL,
    LLMQueryRouter,
    QueryRouter,
    Route,
    RunPlan,
    StaticRouter,
    wants_escalation,
)
from .tool_router import (
    AllToolsSelector,
    LexicalToolSelector,
    ToolSelection,
    ToolSelector,
    filtered_schemas,
)

__all__ = [
    "AllToolsSelector",
    "LexicalToolSelector",
    "ToolSelection",
    "ToolSelector",
    "filtered_schemas",
    "Admission",
    "AgentGateway",
    "ConcurrencyGuard",
    "DIRECT_SYSTEM_PROMPT",
    "ESCALATION_SENTINEL",
    "FORCED_REFLECTION_PROMPT",
    "LLMQueryRouter",
    "QueryRouter",
    "RateLimiter",
    "ReActLoop",
    "REFLECT_AFTER_FAILURE_STATE_KEY",
    "RequestQueue",
    "Route",
    "RunPlan",
    "StateGraph",
    "StaticRouter",
    "ToolDispatcher",
    "is_failure_observation",
    "wants_escalation",
]
