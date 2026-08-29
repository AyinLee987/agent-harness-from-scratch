"""Tool dispatch — single tool execution with retry-once-then-fail semantics.

Extracted from :class:`~agent.ReActAgent` so the trigger layer can dispatch
tool calls independently of the ReAct loop implementation.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..errors import FatalToolError, RecoverableToolError, ToolCallError
from ..observability import get_logger, log_event
from ..state.context import ExecutionContext
from ..tools import ToolRegistry

logger = get_logger(__name__)


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
        correct its call). Explicitly recoverable failures are returned to the
        model. Fatal and unclassified failures abort the current run.
        """
        retry_key = f"retries::{name}"
        retries = ctx.state.get(retry_key, 0)
        started = time.perf_counter()
        argument_keys = sorted(arguments.keys()) if isinstance(arguments, dict) else []
        log_event(
            logger,
            logging.INFO,
            "tool.call.started",
            tool_name=name,
            argument_keys=argument_keys,
            retry=retries,
        )

        def finish(result: str, outcome: str, level: int = logging.INFO) -> str:
            log_event(
                logger,
                level,
                "tool.call.completed",
                tool_name=name,
                outcome=outcome,
                output_chars=len(result),
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            return result

        # Unknown tool — retry once with a hint.
        if name not in self.registry:
            ctx.state[retry_key] = retries + 1
            if retries < self.max_retries:
                return finish((
                    f"ERROR: unknown tool '{name}'. "
                    f"Available tools: {', '.join(self.registry.names())}. "
                    f"Please retry."
                ), "unknown_tool", logging.WARNING)
            return finish(
                f"ERROR: tool '{name}' is unavailable after retry; giving up.",
                "unknown_tool_exhausted",
                logging.WARNING,
            )

        # Malformed arguments — must be a dict.
        if not isinstance(arguments, dict):
            ctx.state[retry_key] = retries + 1
            if retries < self.max_retries:
                return finish((
                    f"ERROR: arguments for '{name}' must be a JSON object. "
                    f"Please retry."
                ), "malformed_arguments", logging.WARNING)
            return finish(
                f"ERROR: malformed arguments for '{name}' after retry; giving up.",
                "malformed_arguments_exhausted",
                logging.WARNING,
            )

        # Dispatch.
        try:
            return finish(self.registry.dispatch(name, arguments), "success")
        except RecoverableToolError as exc:
            return finish(
                f"ERROR calling '{name}': {exc}", "recoverable_error", logging.WARNING
            )
        except FatalToolError:
            log_event(
                logger,
                logging.ERROR,
                "tool.call.failed",
                tool_name=name,
                error_class="fatal",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                exc_info=True,
            )
            raise
        except TypeError as exc:
            ctx.state[retry_key] = retries + 1
            if retries < self.max_retries:
                return finish((
                    f"ERROR calling '{name}': {exc}. "
                    f"Check the arguments and retry."
                ), "invalid_arguments", logging.WARNING)
            return finish(
                f"ERROR calling '{name}': {exc}. Giving up after retry.",
                "invalid_arguments_exhausted",
                logging.WARNING,
            )
        except ToolCallError as exc:
            log_event(
                logger,
                logging.ERROR,
                "tool.call.failed",
                tool_name=name,
                error_class="unrecognized_classified_error",
                error_type=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise FatalToolError(
                f"Tool '{name}' raised an unrecognized classified error: {exc}"
            ) from exc
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "tool.call.failed",
                tool_name=name,
                error_class="unexpected",
                error_type=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                exc_info=True,
            )
            raise FatalToolError(
                f"Tool '{name}' failed unexpectedly: {exc}"
            ) from exc
