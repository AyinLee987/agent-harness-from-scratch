"""Tests for the provider-call deadline and retry policy."""

from __future__ import annotations

import os
import random
import subprocess
import sys

import pytest

from agent.retry import (
    PermanentLLMError,
    RetryPolicy,
    TransientLLMError,
    call_with_retry,
    client_kwargs,
    is_retryable,
    status_of,
)


class _StatusError(Exception):
    """A provider error carrying an HTTP status, the way openai's do."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


class _ResponseStatusError(Exception):
    """Same, but with the status nested on a ``response`` attribute."""

    class _Response:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def __init__(self, status_code: int) -> None:
        self.response = self._Response(status_code)
        super().__init__(f"HTTP {status_code}")


class APITimeoutError(Exception):
    """Name-matched retryable error with no status at all."""


NO_JITTER = RetryPolicy(jitter=0.0, initial_backoff=0.0, max_attempts=3)


def _recorder():
    waits: list[float] = []
    return waits, waits.append


# -- classification ---------------------------------------------------------
def test_status_is_read_from_either_place_providers_put_it():
    assert status_of(_StatusError(429)) == 429
    assert status_of(_ResponseStatusError(503)) == 503
    assert status_of(APITimeoutError()) is None


def test_overload_and_transport_failures_are_retryable():
    for status in (408, 429, 500, 502, 503, 504):
        assert is_retryable(_StatusError(status)), status
    assert is_retryable(APITimeoutError())


def test_request_rejections_are_not_retryable():
    for status in (400, 401, 403, 404, 422):
        assert not is_retryable(_StatusError(status)), status


def test_unknown_exceptions_fail_closed_rather_than_hammering_the_provider():
    assert not is_retryable(ValueError("something else entirely"))


# -- call_with_retry --------------------------------------------------------
def test_a_successful_call_is_not_retried():
    calls = []

    def fn(timeout):
        calls.append(1)
        return "ok"

    assert call_with_retry(fn, policy=NO_JITTER, operation="chat", sleep=lambda _: None) == "ok"
    assert len(calls) == 1


def test_a_retryable_failure_is_retried_until_it_succeeds():
    attempts = []

    def fn(timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise _StatusError(503)
        return "ok"

    result = call_with_retry(fn, policy=NO_JITTER, operation="chat", sleep=lambda _: None)
    assert result == "ok"
    assert len(attempts) == 3


def test_exhausting_the_attempt_budget_raises_transient_with_the_cause_attached():
    attempts = []

    def fn(timeout):
        attempts.append(1)
        raise _StatusError(429)

    with pytest.raises(TransientLLMError) as excinfo:
        call_with_retry(fn, policy=NO_JITTER, operation="chat", sleep=lambda _: None)

    assert len(attempts) == NO_JITTER.max_attempts
    assert isinstance(excinfo.value.__cause__, _StatusError)


def test_a_rejected_request_is_never_retried():
    """Regression guard for the multiplicative-retry trap: a 400 (the shape
    BUGS.md #1 produced) must cost exactly one HTTP request, not three."""

    attempts = []

    def fn(timeout):
        attempts.append(1)
        raise _StatusError(400)

    with pytest.raises(PermanentLLMError):
        call_with_retry(fn, policy=NO_JITTER, operation="chat", sleep=lambda _: None)

    assert len(attempts) == 1


def test_backoff_grows_exponentially_and_is_capped():
    policy = RetryPolicy(
        jitter=0.0, initial_backoff=1.0, backoff_multiplier=2.0, max_backoff=4.0
    )
    assert policy.backoff_for(1) == 0.0
    assert policy.backoff_for(2) == 1.0
    assert policy.backoff_for(3) == 2.0
    assert policy.backoff_for(4) == 4.0
    assert policy.backoff_for(5) == 4.0  # capped


def test_jitter_stays_within_its_declared_fraction():
    policy = RetryPolicy(jitter=0.25, initial_backoff=4.0, backoff_multiplier=1.0)
    rng = random.Random(0)
    for _ in range(50):
        wait = policy.backoff_for(2, rng=rng)
        assert 3.0 <= wait <= 5.0


def test_waits_are_actually_slept_between_attempts():
    waits, record = _recorder()
    policy = RetryPolicy(jitter=0.0, initial_backoff=1.0, max_attempts=3)

    def fn(timeout):
        raise _StatusError(500)

    with pytest.raises(TransientLLMError):
        call_with_retry(fn, policy=policy, operation="chat", sleep=record)

    assert waits == [1.0, 2.0]


def test_the_total_deadline_stops_retrying_even_with_attempts_left():
    """A policy whose backoff alone would blow the deadline must give up at
    that point rather than sleeping past it."""

    policy = RetryPolicy(
        jitter=0.0,
        initial_backoff=100.0,
        max_backoff=100.0,
        max_attempts=5,
        total_deadline_seconds=10.0,
    )
    attempts = []

    def fn(timeout):
        attempts.append(1)
        raise _StatusError(503)

    with pytest.raises(TransientLLMError, match="total deadline"):
        call_with_retry(fn, policy=policy, operation="chat", sleep=lambda _: None)

    assert len(attempts) == 1


def test_invalid_policies_are_rejected_at_construction():
    with pytest.raises(ValueError):
        RetryPolicy(timeout_seconds=0)
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(jitter=1.5)


# -- SDK wiring -------------------------------------------------------------
def test_client_kwargs_disables_the_sdks_own_retries():
    """Leaving the SDK's default of 2 in place would multiply against
    max_attempts (3 x 3 = 9 real requests for one logical call) and hide
    those attempts from this module's logging and total deadline."""

    kwargs = client_kwargs(RetryPolicy(timeout_seconds=42.0))
    assert kwargs == {"timeout": 42.0, "max_retries": 0}


# -- the demo's offline guarantee ------------------------------------------
def test_the_resilience_demo_never_reaches_the_network():
    """``examples/resilience_demo.py`` claims no request leaves the process,
    and constructs a real ``OpenAILLM`` to exercise its actual ``chat()``
    body rather than a copy of it. That combination is only safe while the
    client swap holds, so the claim is guarded here rather than trusted:
    the demo is run in a subprocess with ``connect``/``getaddrinfo``
    blocked, and must still complete.

    Run in a subprocess (matching ``test_mcp.py``'s optional-dependency
    check) because sealing the socket module is not something to do to the
    pytest process itself.
    """

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = """
import sys
# Load the networking stack *first* -- ssl.SSLSocket subclasses
# socket.socket, so sealing it before import breaks the import itself.
import openai, ssl, socket
class Blocked(RuntimeError): pass
def boom(*a, **k):
    raise Blocked("the offline demo tried to reach the network")
socket.socket.connect = boom
socket.socket.connect_ex = boom
socket.create_connection = boom
socket.getaddrinfo = boom

import contextlib, importlib.util, io
spec = importlib.util.spec_from_file_location(
    "demo", "examples/resilience_demo.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
buffer = io.StringIO()
with contextlib.redirect_stdout(buffer):
    module.main()
assert "no circuit breaker exists" in buffer.getvalue()
print("OFFLINE_OK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OFFLINE_OK" in completed.stdout
