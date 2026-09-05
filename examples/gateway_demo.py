"""Example: what the gateway actually protects, and what it doesn't.

    python examples/gateway_demo.py

`AgentGateway` is the single choke point every agent invocation passes
through: sliding-window rate limit -> queue -> concurrency slot. Until
recently it was library code nothing in `app/server.py` referenced, so none
of the behaviour below was actually reaching production (BUGS.md #5).

This exercises each stage against real threads and real clocks, then maps
the rejections onto the HTTP statuses `/api/run` returns. Every "agent" is
a `MockLLM`, so **nothing here talks to a model or opens a socket** -- the
gateway is pure Python with no I/O of its own, and the one endpoint call at
the end has its LLM factory swapped for a local fake.

The last section lists what this design deliberately does not do. A rate
limiter that is described as more than it is becomes a false sense of
safety, which is worse than no limiter at all.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from typing import List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set before anything imports app.server, which calls configure_logging()
# at import time and emits its startup events (tool registration, etc.)
# before this module gets another chance to quiet them. setdefault, so an
# explicit AGENT_LOG_LEVEL still wins if you want to watch the run logs.
os.environ.setdefault("AGENT_LOG_LEVEL", "CRITICAL")

from agent import (
    AgentGateway,
    ConcurrencyLimitExceeded,
    MockLLM,
    QueueTimeout,
    RateLimitExceeded,
    ReActAgent,
    ToolRegistry,
)


def _quiet_agent_logging() -> None:
    """Silence the ``agent`` logger for the duration of the demo.

    Belt and braces alongside the ``AGENT_LOG_LEVEL`` default set at import
    time: ``app.server`` calls ``configure_logging()`` when imported, which
    resets this logger's level, so it is re-applied after that import too.
    """

    logging.getLogger("agent").setLevel(logging.CRITICAL)


def _header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


class _PeakTracker:
    """Counts how many blocks are executing at once, and the high-water mark."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def __enter__(self) -> "_PeakTracker":
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        return self

    def __exit__(self, *_: object) -> None:
        with self._lock:
            self.current -= 1


def _fire(gateway: AgentGateway, count: int, hold: float = 0.0) -> Tuple[int, List[str]]:
    """Send ``count`` requests through the gateway from ``count`` threads.

    Returns (admitted, per-request outcome labels) in submission order.
    """

    outcomes: List[str] = ["?"] * count
    tracker = _PeakTracker()

    def worker(index: int) -> None:
        try:
            with gateway.admit():
                with tracker:
                    time.sleep(hold)
                outcomes[index] = "admitted"
        except RateLimitExceeded:
            outcomes[index] = "429 rate-limited"
        except QueueTimeout:
            outcomes[index] = "503 queue timeout"
        except ConcurrencyLimitExceeded:
            outcomes[index] = "503 no slot"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    gateway_peak = tracker.peak
    admitted = sum(1 for item in outcomes if item == "admitted")
    return admitted, outcomes + [f"__peak__={gateway_peak}"]


# ---------------------------------------------------------------------------
# 1. Sliding-window rate limit
# ---------------------------------------------------------------------------
def demo_rate_limit() -> None:
    _header("1. Rate limit: a burst is shed, and the window really slides")

    gateway = AgentGateway(rate_limit=3, rate_window_seconds=1.0, max_concurrency=10)

    print("  burst of 5 into rate_limit=3 / 1s:")
    for index in range(5):
        try:
            with gateway.admit():
                print(f"    request {index + 1}: admitted")
        except RateLimitExceeded as exc:
            print(f"    request {index + 1}: 429  (Retry-After: {exc.retry_after:.0f}s)")

    print("\n  ...waiting 1.1s for the window to slide...")
    time.sleep(1.1)

    for index in range(3):
        try:
            with gateway.admit():
                print(f"    request {index + 6}: admitted")
        except RateLimitExceeded:
            print(f"    request {index + 6}: 429")

    print(
        "\n  Sliding, not fixed: expired hits are pruned per call, so the budget\n"
        "  refills continuously. A fixed window would let 3 through at 0.99s and\n"
        "  3 more at 1.01s -- 6 in a 20ms span, which is the burst the limit\n"
        "  exists to prevent."
    )


# ---------------------------------------------------------------------------
# 2. Rate limit is checked before queueing
# ---------------------------------------------------------------------------
def demo_order_of_checks() -> None:
    _header("2. Rate limit comes first, on purpose")

    gateway = AgentGateway(
        rate_limit=2, rate_window_seconds=60.0, max_concurrency=8, queue_timeout=5.0
    )
    admitted, outcomes = _fire(gateway, count=6, hold=0.05)

    print(f"  6 requests, rate_limit=2, max_concurrency=8 (plenty of slots free):")
    for index, outcome in enumerate(outcomes[:-1]):
        print(f"    request {index + 1}: {outcome}")
    print(f"\n  admitted {admitted}/6, and nothing sat in the queue: "
          f"queue size = {gateway.queue.size}")
    print(
        "\n  The excess is rejected immediately rather than queued. Queueing a\n"
        "  request that is over the rate limit would convert 'you are asking too\n"
        "  often' into latency for everyone behind it, and the caller would learn\n"
        "  about it queue_timeout seconds later instead of now."
    )


# ---------------------------------------------------------------------------
# 3. Concurrency cap + queueing
# ---------------------------------------------------------------------------
def demo_concurrency() -> None:
    _header("3. Concurrency cap: excess waits rather than failing")

    gateway = AgentGateway(
        rate_limit=1000, rate_window_seconds=60.0, max_concurrency=2, queue_timeout=10.0
    )

    tracker = _PeakTracker()
    hold = 0.3
    count = 6

    def worker() -> None:
        with gateway.admit():
            with tracker:
                time.sleep(hold)

    started = time.monotonic()
    threads = [threading.Thread(target=worker) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    elapsed = time.monotonic() - started

    print(f"  {count} requests x {hold}s each, max_concurrency=2")
    print(f"    peak simultaneous executions : {tracker.peak}  (cap = 2)")
    print(f"    wall clock                   : {elapsed:.2f}s")
    print(f"    unbounded would have taken   : ~{hold:.2f}s")
    print(f"    serialized would have taken  : ~{count * hold:.2f}s")
    print(f"    queue drained                : size = {gateway.queue.size}")
    print(
        "\n  All six completed -- none were rejected. That is the difference\n"
        "  between a concurrency cap and a rate limit: the cap shapes when work\n"
        "  runs, the limit decides whether it runs at all."
    )


# ---------------------------------------------------------------------------
# 4. Queue timeout
# ---------------------------------------------------------------------------
def demo_queue_timeout() -> None:
    _header("4. Waiting has a deadline too")

    gateway = AgentGateway(
        rate_limit=1000, rate_window_seconds=60.0, max_concurrency=1, queue_timeout=0.15
    )
    holding = threading.Event()
    release = threading.Event()

    def hold_the_only_slot() -> None:
        with gateway.admit():
            holding.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=hold_the_only_slot)
    worker.start()
    holding.wait(timeout=5)

    started = time.monotonic()
    try:
        with gateway.admit():
            print("    unexpectedly admitted")
    except QueueTimeout as exc:
        print(f"    503 after {time.monotonic() - started:.2f}s "
              f"(queue_timeout = {gateway.queue.queue_timeout}s)")
        print(f"    {exc}")

    print(f"    queue size after the rejection: {gateway.queue.size}")
    release.set()
    worker.join(timeout=5)
    print(f"    slot returned after the holder finished: "
          f"free = {gateway.concurrency_guard.available}")
    print(
        "\n  A request that waits forever is indistinguishable from a hung one.\n"
        "  Bounding the wait turns an invisible stall into a 503 the caller can\n"
        "  retry or shed."
    )


# ---------------------------------------------------------------------------
# 5. Bookkeeping: the leak that BUGS.md #5 fixed
# ---------------------------------------------------------------------------
def demo_bookkeeping() -> None:
    _header("5. Bookkeeping under load (the leak from BUGS.md #5)")

    gateway = AgentGateway(rate_limit=100000, rate_window_seconds=60.0, max_concurrency=4)
    for _ in range(500):
        with gateway.admit():
            pass

    print(f"  after 500 admitted requests:")
    print(f"    queue size    : {gateway.queue.size}   (was 500 before the fix)")
    print(f"    free slots    : {gateway.concurrency_guard.available} / 4")

    try:
        with gateway.admit():
            raise RuntimeError("the agent run blew up")
    except RuntimeError:
        pass
    print(f"    after a run that raised: free slots = "
          f"{gateway.concurrency_guard.available} / 4, queue = {gateway.queue.size}")
    print(
        "\n  The first version only removed a queue entry on the *timeout* path,\n"
        "  and removed 'the oldest' rather than the one that timed out. An\n"
        "  admitted request was never removed at all, so the list grew by one per\n"
        "  request for the process's lifetime and `size` -- the saturation signal\n"
        "  /api/health reports -- was meaningless. None of it surfaced until the\n"
        "  gateway was first wired into the server."
    )


# ---------------------------------------------------------------------------
# 6. The HTTP mapping
# ---------------------------------------------------------------------------
def demo_http_mapping() -> None:
    _header("6. What a client actually sees (/api/run, MockLLM, no network)")

    import asyncio

    import app.server as server
    from fastapi import HTTPException

    _quiet_agent_logging()  # app.server re-configured it on import

    original_gateway = server.GATEWAY
    original_build = server._build_llm
    server.GATEWAY = AgentGateway(rate_limit=2, rate_window_seconds=60.0)
    server._build_llm = lambda: MockLLM()
    try:
        for index in range(4):
            try:
                result = asyncio.run(
                    server.run(server.RunRequest(task="What is 2 plus 2?"))
                )
                print(f"    request {index + 1}: 200  answer={result.answer[:38]!r}")
            except HTTPException as exc:
                retry_after = (exc.headers or {}).get("Retry-After", "-")
                print(f"    request {index + 1}: {exc.status_code}  "
                      f"Retry-After: {retry_after}  detail={exc.detail}")
    finally:
        server.GATEWAY = original_gateway
        server._build_llm = original_build

    print(
        "\n  429 means 'you are asking too often, back off'; 503 (queue timeout)\n"
        "  means 'the server is saturated, this request never started'. Both\n"
        "  carry Retry-After so a client has something better than a guess.\n"
        "  GET /api/health reports free_slots and queued, which is what tells you\n"
        "  503s are coming before they arrive."
    )


# ---------------------------------------------------------------------------
# 7. What this does not do
# ---------------------------------------------------------------------------
def demo_limitations() -> None:
    _header("7. What this limiter is NOT")

    print(
        "  - Per process, not per deployment. Three replicas behind a load\n"
        "    balancer allow 3 x rate_limit. The backends are pluggable behind\n"
        "    RateLimiter/ConcurrencyGuard precisely so a Redis implementation can\n"
        "    replace them, but the shipped one is in-memory and honest about it.\n"
        "\n"
        "  - Global, not per-caller. There is no tenant or API-key dimension, so\n"
        "    one noisy client can consume the whole budget and everyone else gets\n"
        "    429s caused by someone else. Fixing that means keying the limiter,\n"
        "    not adding a second one.\n"
        "\n"
        "  - A queued request occupies a thread while it waits, because\n"
        "    ConcurrencyGuard.acquire blocks. That is why admission is taken\n"
        "    inside the worker thread rather than the event loop -- documented on\n"
        "    admit() rather than left to be discovered.\n"
        "\n"
        "  - No priority and no cost awareness. A one-step question queues behind\n"
        "    a 100-step research run, and both count as '1' against the rate\n"
        "    limit despite differing by two orders of magnitude in tokens.\n"
        "\n"
        "  - It protects the agent from its callers. It does nothing about the\n"
        "    agent's own calls to a failing provider -- that is a circuit\n"
        "    breaker, which this repo does not have\n"
        "    (see examples/resilience_demo.py, section 8)."
    )


def main() -> None:
    _quiet_agent_logging()
    print("Gateway demo -- MockLLM only, no model calls, no sockets.")
    demo_rate_limit()
    demo_order_of_checks()
    demo_concurrency()
    demo_queue_timeout()
    demo_bookkeeping()
    demo_http_mapping()
    demo_limitations()
    print("\nSections 1-6 are covered by tests/test_gateway.py.\n")


if __name__ == "__main__":
    main()
