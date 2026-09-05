"""Tool dispatch — single tool execution with retry-once-then-fail semantics.

Extracted from :class:`~agent.ReActAgent` so the trigger layer can dispatch
tool calls independently of the ReAct loop implementation.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
from urllib.parse import urlparse

from ..errors import (
    ControlSignal,
    FatalToolError,
    RecoverableToolError,
    ToolCallError,
)
from ..observability import get_logger, log_event
from ..state.context import ExecutionContext
from ..tools import ToolRegistry

logger = get_logger(__name__)


def is_failure_observation(observation: Optional[str]) -> bool:
    """Whether a tool observation string encodes a failure.

    Every non-fatal failure path in :meth:`ToolDispatcher.dispatch` --
    unknown tool, malformed/invalid arguments, ``RecoverableToolError`` --
    returns a plain observation string prefixed with the literal ``"ERROR"``
    rather than raising (only a ``FatalToolError`` sets ``step.error`` and
    aborts the run). This is the one place that convention is spelled out;
    :meth:`~agent.multi_agent.models.SubagentResult.tool_call_summary` and
    the ReAct loop's forced-reflection trigger both call this instead of
    repeating the ``str.startswith("ERROR")`` check inline.
    """
    return (observation or "").startswith("ERROR")


def _source_streak_key(name: str, arguments: dict[str, Any]) -> Optional[str]:
    """Per-(tool, host) key for tracking consecutive failures, or ``None``.

    Only tools called with a ``url`` argument (fetch-like tools) are
    tracked, bucketed by host rather than the full URL: a model retrying a
    blocked site with a different path/query each time never repeats the
    exact same ``(tool, arguments)`` pair, so :func:`_detect_loop`'s own
    documented blind spot ("a loop that varies its arguments each time")
    never fires for it -- this is the narrower, cheaper signal that catches
    that specific case without touching loop detection's semantics.
    """
    url = arguments.get("url") if isinstance(arguments, dict) else None
    if not isinstance(url, str) or not url:
        return None
    host = urlparse(url).netloc.lower() or url
    return f"source_failure_streak::{name}::{host}"


class ToolDispatcher:
    """Execute a single tool call with retry logic and error handling.

    Args:
        registry: The :class:`~agent.ToolRegistry` holding available tools.
        max_retries: Maximum retry attempts for malformed / unknown-tool errors
            (default 1 = try once, then fail).
        source_failure_hint_threshold: Consecutive ``RecoverableToolError``\\ s
            against the same host (see :func:`_source_streak_key`) before a
            hint is appended to the observation telling the model to try a
            different source. ``0`` disables the hint entirely.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_retries: int = 1,
        source_failure_hint_threshold: int = 2,
    ) -> None:
        self.registry = registry
        self.max_retries = max_retries
        self.source_failure_hint_threshold = source_failure_hint_threshold

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

        # Malformed arguments — must be a dict. A raw string arrives here
        # when the provider sent JSON that would not decode: see
        # agent.llm.parse_tool_arguments for why that is passed through
        # rather than flattened to an empty (and therefore *valid*) call.
        if not isinstance(arguments, dict):
            ctx.state[retry_key] = retries + 1
            received = (
                f" The arguments received were not valid JSON: {arguments[:200]!r}."
                if isinstance(arguments, str)
                else f" Received a {type(arguments).__name__}."
            )
            if retries < self.max_retries:
                return finish((
                    f"ERROR: arguments for '{name}' must be a JSON object."
                    f"{received} Please retry with a complete JSON object."
                ), "malformed_arguments", logging.WARNING)
            return finish(
                f"ERROR: malformed arguments for '{name}' after retry; giving up.",
                "malformed_arguments_exhausted",
                logging.WARNING,
            )

        # Dispatch.
        streak_key = _source_streak_key(name, arguments)
        try:
            result = self.registry.dispatch(name, arguments)
        except ControlSignal:
            # Not a failure -- the tool is telling the loop to do something
            # else (today: suspend and wait on jobs). Classifying it as an
            # error would both mislabel it and hide the instruction. See
            # agent/errors.py's ControlSignal.
            log_event(
                logger,
                logging.INFO,
                "tool.call.signalled",
                tool_name=name,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise
        except RecoverableToolError as exc:
            message = f"ERROR calling '{name}': {exc}"
            if streak_key is not None:
                streak = ctx.state.get(streak_key, 0) + 1
                ctx.state[streak_key] = streak
                if streak >= self.source_failure_hint_threshold > 0:
                    message += (
                        f" (This source has failed {streak} times in a row -- it "
                        "may be blocking automated requests, rate-limiting, or "
                        "otherwise unreachable. Try a different source instead of "
                        "retrying this one.)"
                    )
            return finish(message, "recoverable_error", logging.WARNING)
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
        else:
            if streak_key is not None:
                ctx.state[streak_key] = 0
            return finish(result, "success")
