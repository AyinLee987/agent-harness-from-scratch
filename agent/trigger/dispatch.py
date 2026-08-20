"""Tool dispatch — single tool execution with retry-once-then-fail semantics.

Extracted from :class:`~agent.ReActAgent` so the trigger layer can dispatch
tool calls independently of the ReAct loop implementation.
"""

from __future__ import annotations

from typing import Any

from ..state.context import ExecutionContext
from ..tools import ToolRegistry


class ToolDispatcher:
    """Execute a single tool call with retry logic and error handling.

    Args:
        registry: The :class:`~agent.ToolRegistry` holding available tools.
        max_retries: Maximum retry attempts for malformed / unknown-tool errors
            (default 1 = try once, then fail).
    """

    def __init__(self, registry: ToolRegistry, max_retries: int = 1) -> None:
        self.registry = registry
        self.max_retries = max_retries

    def dispatch(
        self, ctx: ExecutionContext, name: str, arguments: dict[str, Any]
    ) -> str:
        """Execute the named tool with ``arguments``.

        Unknown tools and ``TypeError`` are retried up to ``max_retries``
        times (the model gets a helpful error message and a chance to
        correct its call).  All other exceptions are caught and returned
        as error strings without retrying.
        """
        retry_key = f"retries::{name}"
        retries = ctx.state.get(retry_key, 0)

        # Unknown tool — retry once with a hint.
        if name not in self.registry:
            ctx.state[retry_key] = retries + 1
            if retries < self.max_retries:
                return (
                    f"ERROR: unknown tool '{name}'. "
                    f"Available tools: {', '.join(self.registry.names())}. "
                    f"Please retry."
                )
            return f"ERROR: tool '{name}' is unavailable after retry; giving up."

        # Malformed arguments — must be a dict.
        if not isinstance(arguments, dict):
            ctx.state[retry_key] = retries + 1
            if retries < self.max_retries:
                return (
                    f"ERROR: arguments for '{name}' must be a JSON object. "
                    f"Please retry."
                )
            return f"ERROR: malformed arguments for '{name}' after retry; giving up."

        # Dispatch.
        try:
            return self.registry.dispatch(name, arguments)
        except TypeError as exc:
            ctx.state[retry_key] = retries + 1
            if retries < self.max_retries:
                return (
                    f"ERROR calling '{name}': {exc}. "
                    f"Check the arguments and retry."
                )
            return f"ERROR calling '{name}': {exc}. Giving up after retry."
        except Exception as exc:
            return f"ERROR: tool '{name}' raised: {exc}"
