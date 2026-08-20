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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
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
@dataclass
class _QueuedRequest:
    trace_id: str
    enqueued_at: float


class RequestQueue:
    """FIFO request queue with timeout rejection.

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

    def dequeue(self) -> _QueuedRequest | None:
        """Pop the oldest request, or ``None`` if empty."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

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
        trace_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self._rate_limiter = RateLimiter(max_requests=rate_limit)
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
            ConcurrencyLimitExceeded: If all concurrency slots are busy
                and the request cannot be queued.
            QueueTimeout: If the request times out in the queue.
        """
        trace_id = self._trace_id_factory()
        t0 = time.monotonic()

        # 1. Rate limit check.
        if not self._rate_limiter.acquire():
            raise RateLimitExceeded()

        # 2. Enqueue and wait for a concurrency slot.
        queued = self._queue.enqueue(trace_id)
        if not self._concurrency_guard.acquire(blocking=True, timeout=self._queue.queue_timeout):
            self._queue.dequeue()  # best-effort cleanup
            raise QueueTimeout(f"Request {trace_id} timed out in queue")

        queued_ms = (time.monotonic() - queued.enqueued_at) * 1000

        try:
            # 3. Execute.
            if max_steps is not None and hasattr(agent, "max_steps"):
                saved = agent.max_steps
                agent.max_steps = max_steps
                try:
                    result = agent.run(task)
                finally:
                    agent.max_steps = saved
            else:
                result = agent.run(task)

            elapsed_ms = (time.monotonic() - t0) * 1000

            return GatewayResult(
                answer=result.answer,
                success=result.success,
                steps=result.steps,
                tokens=result.tokens,
                stop_reason=result.stop_reason,
                trajectory=result.trajectory,
                trace_id=trace_id,
                queued_ms=queued_ms,
                elapsed_ms=elapsed_ms,
            )
        finally:
            self._concurrency_guard.release()

    # -- context manager ----------------------------------------------------
    def __enter__(self) -> "AgentGateway":
        return self

    def __exit__(self, *_: Any) -> None:
        pass
