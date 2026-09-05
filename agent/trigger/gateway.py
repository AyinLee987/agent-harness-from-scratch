"""Gateway — unified entry point with rate limiting, concurrency control, and queuing.

Provides a single choke point for all agent invocations so that
rate-limiting, concurrency caps, timeouts, and request queuing are
enforced in one place rather than scattered across entry points.

Usage::

    gateway = AgentGateway(rate_limit=100, max_concurrency=10)
    result = gateway.run(agent, "What is 23 times 17?")
    # or async:
    result = await gateway.arun(agent, "What is 23 times 17?")

Design:
    * Pure Python, zero external dependencies (consistent with the project).
    * Pluggable backends: the default in-memory implementation can be swapped
      for Redis / distributed backends behind the same interfaces.
    * Every request gets a ``trace_id`` for end-to-end observability.
"""

from __future__ import annotations

import time
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Admission:
    """One request's accepted place in the gateway.

    Yielded by :meth:`AgentGateway.admit` so a caller that does *not* return
    an :class:`~agent.AgentResult` -- the server's Leader run returns a
    ``MultiAgentRunResult``, which carries subagent results
    :class:`GatewayResult` has no field for -- can still be admission-
    controlled without having its result squeezed through a shape that
    would drop half of it.
    """

    trace_id: str
    queued_ms: float


@dataclass
class GatewayResult:
    """Normalized result from the gateway, wrapping the agent's output."""

    answer: str
    success: bool
    steps: int
    tokens: int
    stop_reason: str
    trajectory: list
    trace_id: str = ""
    queued_ms: float = 0.0
    elapsed_ms: float = 0.0


class GatewayError(Exception):
    """Raised when a request is rejected by the gateway."""


class RateLimitExceeded(GatewayError):
    """Too many requests — retry after the ``retry_after`` window."""

    def __init__(self, retry_after: float = 1.0) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded; retry after {retry_after:.1f}s")


class ConcurrencyLimitExceeded(GatewayError):
    """All execution slots are busy."""


class QueueTimeout(GatewayError):
    """Request sat in the queue too long."""


# ---------------------------------------------------------------------------
# Rate limiter — sliding window
# ---------------------------------------------------------------------------
class RateLimiter:
    """Sliding-window rate limiter.

    Args:
        max_requests: Maximum requests allowed in ``window_seconds``.
        window_seconds: Size of the sliding window in seconds.
    """

    def __init__(self, max_requests: int = 100, window_seconds: float = 1.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: list[float] = []
        self._lock = threading.Lock()

    def acquire(self) -> bool:
        """Try to acquire a rate-limit slot.  Returns ``True`` if allowed."""
        now = time.monotonic()
        with self._lock:
            # Prune expired hits.
            cutoff = now - self.window_seconds
            self._hits = [h for h in self._hits if h > cutoff]
            if len(self._hits) >= self.max_requests:
                return False
            self._hits.append(now)
            return True

    @property
    def current_rate(self) -> float:
        """Approximate current request rate."""
        with self._lock:
            cutoff = time.monotonic() - self.window_seconds
            return sum(1 for h in self._hits if h > cutoff)


# ---------------------------------------------------------------------------
# Concurrency guard — semaphore
# ---------------------------------------------------------------------------
class ConcurrencyGuard:
    """Simple semaphore-based concurrency limiter.

    Args:
        max_concurrency: Maximum number of concurrent executions.
    """

    def __init__(self, max_concurrency: int = 10) -> None:
        self._semaphore = threading.BoundedSemaphore(max_concurrency)

    def acquire(self, blocking: bool = False, timeout: float | None = None) -> bool:
        """Try to acquire a concurrency slot."""
        return self._semaphore.acquire(blocking=blocking, timeout=timeout)

    def release(self) -> None:
        """Release a concurrency slot."""
        self._semaphore.release()

    @property
    def available(self) -> int:
        return self._semaphore._value  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Request queue
# ---------------------------------------------------------------------------
@dataclass(eq=False)
class _QueuedRequest:
    """``eq=False`` so ``list.remove`` matches on identity: two requests that
    happened to be enqueued in the same clock tick must not be confused."""

    trace_id: str
    enqueued_at: float


class RequestQueue:
    """Tracks requests currently waiting for a concurrency slot.

    Each :meth:`enqueue` must be paired with a :meth:`release` of *that*
    handle, whichever way the wait ends. Popping "the oldest" instead is
    what the first version did, and it was wrong twice over: an admitted
    request was never removed at all (so :attr:`size` only ever grew, and
    the list leaked one entry per request for the process's lifetime), and
    the one path that did remove an entry removed a different request's.
    Neither surfaced until the gateway was actually wired into the server.

    Args:
        queue_timeout: Maximum seconds a request can wait before being rejected.
    """

    def __init__(self, queue_timeout: float = 30.0) -> None:
        self.queue_timeout = queue_timeout
        self._queue: list[_QueuedRequest] = []
        self._lock = threading.Lock()

    def enqueue(self, trace_id: str) -> _QueuedRequest:
        """Add a request to the queue.  Returns the queued item."""
        req = _QueuedRequest(trace_id=trace_id, enqueued_at=time.monotonic())
        with self._lock:
            self._queue.append(req)
        return req

    def release(self, req: _QueuedRequest) -> None:
        """Remove ``req`` -- the exact handle :meth:`enqueue` returned."""
        with self._lock:
            try:
                self._queue.remove(req)
            except ValueError:
                # Already pruned by prune_expired(); nothing left to do.
                pass

    def prune_expired(self) -> list[str]:
        """Remove and return trace_ids that have exceeded the timeout."""
        now = time.monotonic()
        expired: list[str] = []
        with self._lock:
            kept: list[_QueuedRequest] = []
            for req in self._queue:
                if now - req.enqueued_at > self.queue_timeout:
                    expired.append(req.trace_id)
                else:
                    kept.append(req)
            self._queue = kept
        return expired

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._queue)


# ---------------------------------------------------------------------------
# Agent Gateway
# ---------------------------------------------------------------------------
class AgentGateway:
    """Unified entry point for agent invocations.

    Enforces rate limiting → concurrency control → queuing, then
    dispatches to the agent's ``run()`` method.

    Args:
        rate_limit: Max requests per second (sliding window).
        max_concurrency: Max simultaneous agent executions.
        queue_timeout: Max seconds a request waits in queue.
        trace_id_factory: Optional callable to generate trace ids.

    Example::

        from agent import MockLLM, ReActAgent, ToolRegistry, tool
        from agent.trigger import AgentGateway

        @tool
        def echo(text: str) -> str: return text

        agent = ReActAgent(llm=MockLLM(), tools=ToolRegistry([echo]))
        gateway = AgentGateway(rate_limit=100, max_concurrency=10)

        result = gateway.run(agent, "Hello!")
        print(result.answer)
    """

    def __init__(
        self,
        rate_limit: int = 100,
        max_concurrency: int = 10,
        queue_timeout: float = 30.0,
        rate_window_seconds: float = 1.0,
        trace_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._rate_limiter = RateLimiter(
            max_requests=rate_limit, window_seconds=rate_window_seconds
        )
        self._concurrency_guard = ConcurrencyGuard(max_concurrency)
        self._queue = RequestQueue(queue_timeout=queue_timeout)
        self._trace_id_factory = trace_id_factory or (lambda: uuid.uuid4().hex[:12])

    # -- properties ---------------------------------------------------------
    @property
    def rate_limiter(self) -> RateLimiter:
        return self._rate_limiter

    @property
    def concurrency_guard(self) -> ConcurrencyGuard:
        return self._concurrency_guard

    @property
    def queue(self) -> RequestQueue:
        return self._queue

    # -- admission ----------------------------------------------------------
    @contextmanager
    def admit(self, trace_id: Optional[str] = None) -> Iterator[Admission]:
        """Acquire a place to run, and hold it for the duration of the block.

        This is the gateway's real primitive; :meth:`run` is a convenience
        wrapper for the common "run a plain agent" case. Callers whose unit
        of work isn't a bare ``agent.run(task)`` -- the server runs a Leader
        through a :class:`~agent.MultiAgentOrchestrator` and gets back a
        result type with subagent data on it -- use this directly and keep
        their own result shape.

        The wait for a concurrency slot is a **blocking** one, so this must
        be entered from a worker thread, never from an event loop. That
        trade-off is deliberate: it keeps the gateway a plain, dependency-
        free synchronous component that a Redis-backed distributed
        implementation could replace behind the same interface, at the cost
        of a queued request occupying a thread while it waits.

        Raises:
            RateLimitExceeded: If the rate limit is hit.
            QueueTimeout: If the request waits longer than ``queue_timeout``
                for a concurrency slot.
        """

        trace_id = trace_id or self._trace_id_factory()

        if not self._rate_limiter.acquire():
            raise RateLimitExceeded(retry_after=self._rate_limiter.window_seconds)

        queued = self._queue.enqueue(trace_id)
        try:
            if not self._concurrency_guard.acquire(
                blocking=True, timeout=self._queue.queue_timeout
            ):
                raise QueueTimeout(f"Request {trace_id} timed out in queue")
        finally:
            # Paired with enqueue() on every exit path, including the
            # timeout raise -- see RequestQueue's docstring for the leak
            # this replaces.
            self._queue.release(queued)

        try:
            yield Admission(
                trace_id=trace_id,
                queued_ms=(time.monotonic() - queued.enqueued_at) * 1000,
            )
        finally:
            self._concurrency_guard.release()

    # -- synchronous entry --------------------------------------------------
    def run(self, agent: Any, task: str, *, max_steps: int | None = None) -> GatewayResult:
        """Run a task through the gateway synchronously.

        Args:
            agent: A :class:`~agent.ReActAgent` instance (or any object with
                a ``run(task) -> AgentResult`` method).
            task: The task string to pass to the agent.
            max_steps: Optional per-request step override.

        Returns:
            A :class:`GatewayResult` wrapping the agent's output.

        Raises:
            RateLimitExceeded: If the rate limit is hit.
            QueueTimeout: If the request times out in the queue.
        """
        t0 = time.monotonic()
        with self.admit() as admission:
            if max_steps is not None and hasattr(agent, "max_steps"):
                saved = agent.max_steps
                agent.max_steps = max_steps
                try:
                    result = agent.run(task)
                finally:
                    agent.max_steps = saved
            else:
                result = agent.run(task)

            return GatewayResult(
                answer=result.answer,
                success=result.success,
                steps=result.steps,
                tokens=result.tokens,
                stop_reason=result.stop_reason,
                trajectory=result.trajectory,
                trace_id=admission.trace_id,
                queued_ms=admission.queued_ms,
                elapsed_ms=(time.monotonic() - t0) * 1000,
            )

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "AgentGateway":
        return self

    def __exit__(self, *_: Any) -> None:
        pass
