"""Deadline and retry policy for provider API calls.

Every provider call in this repo used to inherit whatever the vendor SDK
defaulted to. For the ``openai`` SDK that is a **600-second** timeout and
**two blind retries**, which produces three bad behaviours the agent loop
has no way to see or recover from:

* a hung provider holds one ReAct step for ten minutes, while the run's own
  ``max_steps``/``max_tokens`` guardrails -- which count steps and tokens,
  not seconds -- sit there with nothing to trip on;
* a permanently-broken request (a malformed transcript, a revoked key) is
  retried anyway, turning one fast ``400`` into three slow ones;
* nothing is logged between "call started" and "call failed", so a run that
  succeeded only after two internal retries is indistinguishable from one
  that succeeded first try.

This module makes all three explicit. Callers own a :class:`RetryPolicy`,
pass it to :func:`call_with_retry`, and get back either a value or one of
two errors, mirroring the tool taxonomy in :mod:`agent.errors`:

* :class:`TransientLLMError` -- the provider was unreachable, overloaded, or
  rate-limiting, and the attempt budget ran out. The request itself was
  fine; the same call might well work later.
* :class:`PermanentLLMError` -- the provider rejected the request itself.
  Retrying cannot help, so it is never retried.

Deliberately dependency-free: classification duck-types on the exception's
name and HTTP status rather than importing ``openai``, because ``openai``
is an optional dependency here (the repo must stay importable, and CI must
stay green, with no provider SDK installed at all).
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar

from .observability import get_logger, log_event

T = TypeVar("T")

logger = get_logger(__name__)


class LLMCallError(Exception):
    """Base class for a provider call that could not be completed."""


class TransientLLMError(LLMCallError):
    """The provider was unavailable/overloaded and the retry budget ran out.

    The request was well-formed -- an identical call may succeed later.
    """


class PermanentLLMError(LLMCallError):
    """The provider rejected the request itself; retrying cannot help."""


#: HTTP statuses worth another attempt: transport hiccups, contention, and
#: the provider's own overload signals.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: HTTP statuses that mean the request is wrong. ``400`` is the one that
#: matters most in practice: BUGS.md #1 is exactly this shape -- an orphaned
#: ``tool`` message produced a ``400`` that no number of retries would fix.
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 405, 413, 422})

#: Exception class names (``openai`` and ``httpx`` alike) that are retryable
#: even when they carry no HTTP status -- a connection that never opened has
#: no response to read a status off.
_RETRYABLE_NAMES = frozenset({
    "APIConnectionError",
    "APITimeoutError",
    "APIConnectionTimeoutError",
    "ConnectError",
    "ConnectTimeout",
    "InternalServerError",
    "RateLimitError",
    "ReadTimeout",
    "RemoteProtocolError",
    "TimeoutException",
    "WriteTimeout",
})


@dataclass(frozen=True)
class RetryPolicy:
    """How long one provider call may take, and how often it may be retried.

    Args:
        timeout_seconds: Per-attempt deadline handed to the provider SDK.
            This is the number that replaces the SDK's 600-second default;
            it must be short enough that a hung provider trips it well
            before a human gives up on the request.
        max_attempts: Total attempts including the first (``1`` disables
            retrying entirely).
        initial_backoff: Seconds to wait before the second attempt.
        backoff_multiplier: Growth factor applied after each failed attempt.
        max_backoff: Ceiling on a single wait.
        jitter: Fraction of the computed wait to randomize by, so a fleet of
            agents retrying after the same outage does not resynchronize
            into a thundering herd. ``0`` makes waits deterministic (what
            the tests use).
        total_deadline_seconds: Optional wall-clock ceiling across *all*
            attempts. Without it, ``max_attempts`` x ``timeout_seconds``
            plus backoff is the real worst case, which is usually far longer
            than the caller expects. ``None`` disables the ceiling.
    """

    timeout_seconds: float = 60.0
    max_attempts: int = 3
    initial_backoff: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff: float = 8.0
    jitter: float = 0.25
    total_deadline_seconds: Optional[float] = 180.0

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1.")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be between 0 and 1.")

    def backoff_for(self, attempt: int, *, rng: Optional[random.Random] = None) -> float:
        """Seconds to wait before attempt number ``attempt`` (1-based)."""

        if attempt < 2:
            return 0.0
        raw = self.initial_backoff * (self.backoff_multiplier ** (attempt - 2))
        capped = min(raw, self.max_backoff)
        if not self.jitter:
            return capped
        spread = capped * self.jitter
        return max(0.0, capped + (rng or random).uniform(-spread, spread))


def status_of(exc: BaseException) -> Optional[int]:
    """Best-effort HTTP status for a provider exception, or ``None``.

    Providers put it in one of two places depending on SDK version, and a
    connection error has neither -- hence the duck-typing rather than an
    ``isinstance`` chain against a package that may not be installed.
    """

    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def is_retryable(exc: BaseException) -> bool:
    """Whether ``exc`` is worth another attempt.

    Unknown exceptions are treated as **not** retryable, matching the
    dispatcher's fail-closed stance in :mod:`agent.trigger.dispatch`: an
    unclassified failure is a bug to classify, not a reason to hammer the
    provider three more times.
    """

    status = status_of(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    return type(exc).__name__ in _RETRYABLE_NAMES


def call_with_retry(
    fn: Callable[[float], T],
    *,
    policy: RetryPolicy,
    operation: str,
    sleep: Callable[[float], None] = time.sleep,
    rng: Optional[random.Random] = None,
) -> T:
    """Call ``fn`` under ``policy``, retrying only what is worth retrying.

    Args:
        fn: The provider call. It receives **this attempt's timeout in
            seconds** as its only argument and must hand that to the
            provider (the ``openai`` SDK takes a per-request ``timeout=``).
            Only the SDK can enforce a deadline on a socket it owns, so
            this module computes the budget and the caller applies it.
        policy: The governing :class:`RetryPolicy`.
        operation: Short name for logging, e.g. ``"chat"`` or ``"embed"``.
        sleep: Injectable for tests; defaults to :func:`time.sleep`.
        rng: Injectable jitter source for tests.

    Returns:
        Whatever ``fn`` returned.

    Raises:
        PermanentLLMError: The provider rejected the request itself.
        TransientLLMError: Retries were exhausted, or the total deadline
            passed, while failing on retryable errors.

    ``total_deadline_seconds`` is a real ceiling on wall-clock time, not
    just on sleeping. It used to be checked in one place only -- against the
    next backoff -- so an attempt that itself ran long could sail past it:
    with a 10s deadline, a 9s first attempt followed by a 0.1s backoff and a
    9s second attempt took 18s and the late result was accepted. Each
    attempt is now given ``min(timeout_seconds, remaining)``, and an
    exhausted budget stops the loop before the call is made. See BUGS.md #15.
    """

    started = time.monotonic()
    last: Optional[BaseException] = None

    for attempt in range(1, policy.max_attempts + 1):
        if attempt > 1:
            wait = policy.backoff_for(attempt, rng=rng)
            remaining = _remaining(policy, started)
            if remaining is not None and remaining <= wait:
                raise TransientLLMError(
                    f"LLM {operation} gave up after {attempt - 1} attempt(s): "
                    f"total deadline of {policy.total_deadline_seconds}s reached "
                    f"({last})"
                ) from last
            log_event(
                logger,
                logging.WARNING,
                "llm.call.retrying",
                operation=operation,
                attempt=attempt,
                max_attempts=policy.max_attempts,
                backoff_seconds=round(wait, 3),
                error_type=type(last).__name__ if last else None,
                status_code=status_of(last) if last else None,
            )
            sleep(wait)

        attempt_timeout = policy.timeout_seconds
        remaining = _remaining(policy, started)
        if remaining is not None:
            if remaining <= 0:
                raise TransientLLMError(
                    f"LLM {operation} gave up after {attempt - 1} attempt(s): "
                    f"total deadline of {policy.total_deadline_seconds}s reached "
                    f"({last})"
                ) from last
            # Never let one attempt outlive the budget for the whole call.
            attempt_timeout = min(attempt_timeout, remaining)

        try:
            return fn(attempt_timeout)
        except Exception as exc:
            last = exc
            if not is_retryable(exc):
                log_event(
                    logger,
                    logging.ERROR,
                    "llm.call.rejected",
                    operation=operation,
                    attempt=attempt,
                    error_type=type(exc).__name__,
                    status_code=status_of(exc),
                )
                raise PermanentLLMError(
                    f"LLM {operation} was rejected and will not be retried: {exc}"
                ) from exc

    log_event(
        logger,
        logging.ERROR,
        "llm.call.exhausted",
        operation=operation,
        attempts=policy.max_attempts,
        elapsed_ms=round((time.monotonic() - started) * 1000, 2),
        error_type=type(last).__name__ if last else None,
        status_code=status_of(last) if last else None,
    )
    raise TransientLLMError(
        f"LLM {operation} failed after {policy.max_attempts} attempt(s): {last}"
    ) from last


def _remaining(policy: RetryPolicy, started: float) -> Optional[float]:
    if policy.total_deadline_seconds is None:
        return None
    return policy.total_deadline_seconds - (time.monotonic() - started)


def client_kwargs(policy: RetryPolicy) -> Dict[str, Any]:
    """SDK constructor kwargs that hand deadline control to ``policy``.

    ``max_retries=0`` is the load-bearing half: leaving the SDK's own
    default of 2 in place would multiply against ``max_attempts`` here
    (3 x 3 = 9 real HTTP requests for one logical call), and the SDK's
    retries are invisible to :func:`call_with_retry`'s logging and to the
    total deadline.

    The ``timeout`` here is only the client-wide default. Each individual
    attempt overrides it with the tighter of ``timeout_seconds`` and the
    remaining total budget -- see :func:`call_with_retry`.
    """

    return {"timeout": policy.timeout_seconds, "max_retries": 0}
