"""Tests for gateway admission control and its wiring into the server."""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from agent import (
    AgentGateway,
    MockLLM,
    QueueTimeout,
    RateLimitExceeded,
    ReActAgent,
    ToolRegistry,
)
from agent.trigger.gateway import RequestQueue


# -- RequestQueue bookkeeping ----------------------------------------------
def test_admitted_requests_are_removed_from_the_queue():
    """Regression guard: the first version enqueued every request and only
    ever removed one on the timeout path, so the list grew by one entry per
    request forever and `size` never went back to zero."""

    queue = RequestQueue()
    first = queue.enqueue("a")
    second = queue.enqueue("b")
    assert queue.size == 2

    queue.release(first)
    queue.release(second)
    assert queue.size == 0


def test_releasing_removes_that_request_and_not_merely_the_oldest():
    queue = RequestQueue()
    first = queue.enqueue("a")
    second = queue.enqueue("b")

    queue.release(second)
    assert queue.size == 1
    queue.release(first)
    assert queue.size == 0


def test_releasing_twice_is_harmless():
    queue = RequestQueue()
    req = queue.enqueue("a")
    queue.release(req)
    queue.release(req)
    assert queue.size == 0


# -- admit() ----------------------------------------------------------------
def test_admit_yields_a_trace_id_and_frees_the_slot_afterwards():
    gateway = AgentGateway(max_concurrency=1)

    with gateway.admit() as admission:
        assert admission.trace_id
        assert gateway.concurrency_guard.available == 0
        assert gateway.queue.size == 0

    assert gateway.concurrency_guard.available == 1


def test_admit_releases_its_slot_even_when_the_body_raises():
    gateway = AgentGateway(max_concurrency=1)

    with pytest.raises(RuntimeError):
        with gateway.admit():
            raise RuntimeError("boom")

    assert gateway.concurrency_guard.available == 1
    assert gateway.queue.size == 0


def test_the_rate_limit_rejects_with_a_retry_after_window():
    gateway = AgentGateway(rate_limit=1, rate_window_seconds=60.0)

    with gateway.admit():
        pass

    with pytest.raises(RateLimitExceeded) as excinfo:
        with gateway.admit():
            pass

    assert excinfo.value.retry_after == 60.0


def test_a_request_that_cannot_get_a_slot_in_time_is_rejected_and_not_leaked():
    gateway = AgentGateway(max_concurrency=1, queue_timeout=0.05)
    holding = threading.Event()
    release = threading.Event()

    def hold():
        with gateway.admit():
            holding.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=hold)
    worker.start()
    assert holding.wait(timeout=5)

    try:
        with pytest.raises(QueueTimeout):
            with gateway.admit():
                pass
        assert gateway.queue.size == 0
    finally:
        release.set()
        worker.join(timeout=5)

    assert gateway.concurrency_guard.available == 1


def test_run_still_wraps_a_plain_agent_through_admit():
    gateway = AgentGateway()
    agent = ReActAgent(llm=MockLLM(), tools=ToolRegistry([]))

    free_before = gateway.concurrency_guard.available
    result = gateway.run(agent, "hello")

    assert result.trace_id
    assert result.answer
    assert gateway.concurrency_guard.available == free_before
    assert gateway.queue.size == 0


# -- server wiring ----------------------------------------------------------
def test_run_endpoint_returns_429_when_the_gateway_rate_limits(monkeypatch):
    """The gateway used to be entirely unreferenced by app/server.py -- this
    asserts a request actually passes through it."""

    from fastapi import HTTPException

    import app.server as server

    gateway = AgentGateway(rate_limit=1, rate_window_seconds=60.0)
    monkeypatch.setattr(server, "GATEWAY", gateway)
    monkeypatch.setattr(server, "_build_llm", lambda: MockLLM())

    first = asyncio.run(server.run(server.RunRequest(task="What is 2 plus 2?")))
    assert first.answer

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(server.run(server.RunRequest(task="again")))

    assert excinfo.value.status_code == 429
    assert excinfo.value.headers["Retry-After"] == "60"


def test_health_reports_gateway_saturation(monkeypatch):
    import app.server as server

    monkeypatch.setattr(server, "GATEWAY", AgentGateway(max_concurrency=4))
    monkeypatch.setattr(server, "_build_llm", lambda: MockLLM())

    payload = asyncio.run(server.health())

    assert payload["gateway"]["free_slots"] == 4
    assert payload["gateway"]["queued"] == 0


def test_a_disabled_gateway_admits_everything(monkeypatch):
    import app.server as server

    monkeypatch.setattr(server, "GATEWAY", None)
    monkeypatch.setattr(server, "_build_llm", lambda: MockLLM())

    for _ in range(3):
        result = asyncio.run(server.run(server.RunRequest(task="What is 2 plus 2?")))
        assert result.answer

    assert "gateway" not in asyncio.run(server.health())
