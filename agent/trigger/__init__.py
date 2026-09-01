"""Trigger Layer — decides WHEN and HOW the agent acts.

Components:
    - Gateway: rate limiting, concurrency control, request queuing
    - StateGraph: generic directed graph execution engine
    - ReActLoop: the think → act → observe cycle
    - Dispatch: tool execution with retry and error handling
"""

from .gateway import AgentGateway, ConcurrencyGuard, RateLimiter, RequestQueue
from .graph import StateGraph
from .react_loop import (
    FORCED_REFLECTION_PROMPT,
    REFLECT_AFTER_FAILURE_STATE_KEY,
    ReActLoop,
)
from .dispatch import ToolDispatcher, is_failure_observation

__all__ = [
    "AgentGateway",
    "ConcurrencyGuard",
    "FORCED_REFLECTION_PROMPT",
    "RateLimiter",
    "ReActLoop",
    "REFLECT_AFTER_FAILURE_STATE_KEY",
    "RequestQueue",
    "StateGraph",
    "ToolDispatcher",
    "is_failure_observation",
]
