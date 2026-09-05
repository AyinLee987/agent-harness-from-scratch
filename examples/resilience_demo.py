"""Example: what happens to a run when the model provider misbehaves.

    python examples/resilience_demo.py

Exercises every provider-failure path in `agent/retry.py` and the ReAct
loop's degradation around it -- classification, backoff, the total
deadline, the SDK double-retry trap, and what a run keeps when the provider
goes away mid-trajectory.

**Nothing here talks to a real model.** Three independent guarantees, so a
mistake in one still can't leak a prompt:

1. every "LLM" is a local fake -- either a `MockLLM` subclass or a stub
   object standing in for `client.chat.completions`;
2. the one place a real `OpenAILLM` is constructed (to exercise its actual
   `chat()` code path rather than a reimplementation of it) has its
   `_client` replaced before any call, and gets a throwaway key;
3. that same client is pointed at `127.0.0.1:9` -- the discard port -- so
   even if guarantee 2 somehow failed, the request could not leave the
   machine.

The last section is the important one: it demonstrates a mechanism this
repo does **not** have. There is no circuit breaker -- no failure-rate
tracking, no open/half-open state, nothing that stops calling a provider
that has been failing for the last five minutes. Every run pays the full
retry budget independently. The section measures that cost rather than
asserting it.
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import (
    AgentGateway,
    MockLLM,
    QueueTimeout,
    RateLimitExceeded,
    ReActAgent,
    RetryPolicy,
    ToolRegistry,
    tool,
)
from agent.llm import LLMResponse, ToolCall, Usage
from agent.retry import (
    PermanentLLMError,
    TransientLLMError,
    call_with_retry,
    client_kwargs,
    is_retryable,
)

DISCARD_PORT_URL = "http://127.0.0.1:9/v1"


# ---------------------------------------------------------------------------
# Local stand-ins for a provider. None of these open a socket.
# ---------------------------------------------------------------------------
class ProviderError(Exception):
    """Shaped like an ``openai`` error: carries an HTTP status."""

    def __init__(self, status_code: int, message: str = "") -> None:
        self.status_code = status_code
        super().__init__(message or f"HTTP {status_code}")


class ConnectTimeout(Exception):
    """Shaped like a transport error: no status to read at all."""


class _FakeCompletions:
    """Stands in for ``client.chat.completions``.

    ``script`` is a list of either exceptions to raise or strings to return,
    consumed one per call, so a test can say "fail twice, then succeed".
    """

    def __init__(self, script: List[Any]) -> None:
        self.script = list(script)
        self.calls = 0

    def create(self, **_: Any) -> Any:
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return _response(item)


def _response(text: str) -> Any:
    """A minimal object with the shape ``OpenAILLM.chat`` reads."""

    class _Message:
        content = text
        tool_calls: List[Any] = []

    class _Choice:
        message = _Message()

    class _Usage:
        prompt_tokens = 7
        completion_tokens = 5

    class _Completion:
        choices = [_Choice()]
        usage = _Usage()

    return _Completion()


class _FakeClient:
    def __init__(self, script: List[Any]) -> None:
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(script)})()

    @property
    def calls(self) -> int:
        return self.chat.completions.calls


def _offline_llm(script: List[Any], policy: RetryPolicy):
    """A real ``OpenAILLM`` whose client is a local fake.

    Constructing the real class matters: this exercises the actual
    ``chat()`` body, including its ``call_with_retry`` wrapping, rather
    than a demo-only reimplementation that could drift from it.
    """

    from agent import OpenAILLM

    os.environ.setdefault("OPENAI_API_KEY", "sk-offline-demo-not-a-real-key")
    llm = OpenAILLM(base_url=DISCARD_PORT_URL, retry_policy=policy)
    llm._client = _FakeClient(script)  # noqa: SLF001 - deliberate, see docstring
    return llm


def _header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. Classification
# ---------------------------------------------------------------------------
def demo_classification() -> None:
    _header("1. Which failures are worth retrying")

    cases = [
        (ProviderError(429, "rate limited"), "429 rate limit"),
        (ProviderError(503, "overloaded"), "503 service unavailable"),
        (ConnectTimeout("no route to host"), "transport timeout (no status)"),
        (ProviderError(400, "malformed tool message"), "400 bad request"),
        (ProviderError(401, "bad key"), "401 unauthorized"),
        (ValueError("something else entirely"), "unrecognized exception"),
    ]
    for exc, label in cases:
        verdict = "retry" if is_retryable(exc) else "raise immediately"
        print(f"  {label:<32} -> {verdict}")

    print(
        "\n  An unrecognized exception is NOT retried: that matches the tool\n"
        "  dispatcher's fail-closed stance -- an unclassified failure is a bug\n"
        "  to classify, not a reason to hammer the provider three more times."
    )


# ---------------------------------------------------------------------------
# 2. Backoff
# ---------------------------------------------------------------------------
def demo_backoff() -> None:
    _header("2. Backoff schedule (no real sleeping)")

    waits: List[float] = []
    policy = RetryPolicy(jitter=0.0, initial_backoff=0.5, max_backoff=4.0, max_attempts=6)

    def always_503(timeout: float) -> str:
        raise ProviderError(503)

    try:
        call_with_retry(
            always_503, policy=policy, operation="chat", sleep=waits.append
        )
    except TransientLLMError:
        pass
    print(f"  jitter=0.0 -> waits: {waits}  (doubling, capped at {policy.max_backoff}s)")

    jittered = RetryPolicy(jitter=0.25, initial_backoff=4.0, backoff_multiplier=1.0)
    rng = random.Random(7)
    samples = [round(jittered.backoff_for(2, rng=rng), 2) for _ in range(6)]
    print(f"  jitter=0.25 -> waits around 4.0s: {samples}")
    print(
        "\n  Jitter is not decoration: without it every agent that failed during\n"
        "  the same outage retries on the same schedule and re-synchronizes into\n"
        "  a thundering herd the moment the provider comes back."
    )


# ---------------------------------------------------------------------------
# 3. Total deadline
# ---------------------------------------------------------------------------
def demo_total_deadline() -> None:
    _header("3. The total deadline beats the attempt count")

    attempts = 0

    def always_503(timeout: float) -> str:
        nonlocal attempts
        attempts += 1
        raise ProviderError(503)

    policy = RetryPolicy(
        jitter=0.0,
        initial_backoff=100.0,
        max_backoff=100.0,
        max_attempts=5,
        total_deadline_seconds=10.0,
    )
    try:
        call_with_retry(
            always_503, policy=policy, operation="chat", sleep=lambda _: None
        )
    except TransientLLMError as exc:
        print(f"  max_attempts=5, but stopped after {attempts} attempt(s)")
        print(f"  reason: {exc}")
    print(
        "\n  Without this ceiling the real worst case is\n"
        "  max_attempts x timeout_seconds + backoff -- far longer than the\n"
        "  caller believes it configured."
    )


# ---------------------------------------------------------------------------
# 4. The double-retry trap
# ---------------------------------------------------------------------------
def demo_sdk_retries() -> None:
    _header("4. The SDK's own retries are switched off")

    kwargs = client_kwargs(RetryPolicy(timeout_seconds=60.0, max_attempts=3))
    print(f"  kwargs handed to OpenAI(...): {kwargs}")
    print(
        "\n  The openai SDK defaults to timeout=600.0, max_retries=2.\n"
        "  Leaving those in place would mean:\n"
        "    - a hung provider holds one ReAct step for 10 minutes, and\n"
        "      neither max_steps nor max_tokens can catch it (they count\n"
        "      steps and tokens, not seconds);\n"
        "    - 3 attempts here x 2 SDK retries = 9 real HTTP requests for one\n"
        "      logical call, six of them invisible to this module's logging\n"
        "      and to the total deadline."
    )


# ---------------------------------------------------------------------------
# 5. The real OpenAILLM.chat path, offline
# ---------------------------------------------------------------------------
def demo_real_client_path() -> None:
    _header("5. The real OpenAILLM.chat() code path (fake client, no network)")

    policy = RetryPolicy(jitter=0.0, initial_backoff=0.0, max_attempts=3)
    messages = [{"role": "user", "content": "(never leaves this process)"}]

    llm = _offline_llm(
        [ProviderError(429), ProviderError(503), "recovered on the third attempt"],
        policy,
    )
    print(f"  class under test : {type(llm).__name__} (the real one)")
    print(f"  client           : {type(llm._client).__name__} (local fake)")
    print(f"  base_url         : {DISCARD_PORT_URL} (discard port, refuses)")
    response = llm.chat(messages)
    print(f"  429 -> 503 -> ok : {llm._client.calls} HTTP attempts, "
          f"answer={response.content!r}")

    llm = _offline_llm([ProviderError(400, "orphaned tool message")], policy)
    try:
        llm.chat(messages)
    except PermanentLLMError as exc:
        print(f"  400              : {llm._client.calls} HTTP attempt, raised {type(exc).__name__}")
        print(f"                     {exc}")

    llm = _offline_llm([ProviderError(503)], policy)
    try:
        llm.chat(messages)
    except TransientLLMError:
        print(f"  503 throughout   : {llm._client.calls} HTTP attempts, then TransientLLMError")

    print(
        "\n  The 400 costs exactly one request. That is the shape of BUGS.md #1\n"
        "  (an orphaned `tool` message) -- deterministic, so retrying it turned\n"
        "  one fast failure into three slow ones."
    )


# ---------------------------------------------------------------------------
# 6. Degradation inside the loop
# ---------------------------------------------------------------------------
def demo_loop_degradation() -> None:
    _header("6. What a run keeps when the provider disappears mid-trajectory")

    @tool
    def lookup(city: str) -> str:
        """Look up a city's population."""
        return f"{city}: 21,540,000"

    class _DiesAfterOneToolCall(MockLLM):
        calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(id="a", name="lookup", arguments={"city": "Beijing"})
                    ],
                    usage=Usage(120, 30),
                )
            raise TransientLLMError("provider unavailable after 3 attempts")

    result = ReActAgent(
        llm=_DiesAfterOneToolCall(), tools=ToolRegistry([lookup])
    ).run("How many people live in Beijing?")

    print(f"  stop_reason : {result.stop_reason}")
    print(f"  success     : {result.success}")
    print(f"  steps kept  : {result.steps}")
    print(f"  tokens kept : {result.tokens}")
    print(f"  observation : {result.trajectory[0]['observation']!r}")
    print(f"  answer      : {result.answer!r}")
    print(
        "\n  Before this, the exception propagated out of run() and all of the\n"
        "  above was discarded -- the caller learned only that something failed.\n"
        "  A PermanentLLMError still propagates on purpose: a request the\n"
        "  provider rejects outright is a defect to fix, and degrading around it\n"
        "  would hide it behind a plausible-looking partial answer."
    )


# ---------------------------------------------------------------------------
# 7. Load shedding at the gateway
# ---------------------------------------------------------------------------
def demo_load_shedding() -> None:
    _header("7. Shedding load at the door (the closest thing to a breaker)")

    gateway = AgentGateway(rate_limit=2, rate_window_seconds=60.0, max_concurrency=1)
    admitted = 0
    for index in range(5):
        try:
            with gateway.admit():
                admitted += 1
        except RateLimitExceeded as exc:
            print(f"  request {index + 1}: 429, retry after {exc.retry_after:.0f}s")
            continue
        except QueueTimeout:
            print(f"  request {index + 1}: 503, timed out queueing")
            continue
        print(f"  request {index + 1}: admitted")
    print(f"\n  admitted {admitted}/5; queue drained cleanly: size={gateway.queue.size}")
    print(
        "\n  This protects the agent from its callers. It does nothing about the\n"
        "  agent's own calls to a failing provider -- which is the next section."
    )


# ---------------------------------------------------------------------------
# 8. The gap: there is no circuit breaker
# ---------------------------------------------------------------------------
def demo_missing_circuit_breaker() -> None:
    _header("8. THE GAP: no circuit breaker exists")

    policy = RetryPolicy(jitter=0.0, initial_backoff=0.0, max_attempts=3)
    total_http = 0
    started = time.monotonic()

    for run_index in range(10):
        llm = _offline_llm([ProviderError(503)], policy)
        try:
            llm.chat([{"role": "user", "content": "(never leaves this process)"}])
        except TransientLLMError:
            pass
        total_http += llm._client.calls

    elapsed = time.monotonic() - started
    print(f"  10 consecutive runs against a provider that is fully down:")
    print(f"    HTTP attempts made : {total_http}   (every run paid the full budget)")
    print(f"    attempts avoided   : 0")
    print(f"    wall clock         : {elapsed:.3f}s here, because backoff is stubbed to 0")
    print(
        f"\n  With the shipped defaults (initial_backoff=0.5, multiplier=2.0) each\n"
        f"  of those runs would have slept ~1.5s before failing, on top of up to\n"
        f"  3 x 60s of connection timeouts. Run 10 knows nothing about runs 1-9."
    )
    print(
        "\n  A circuit breaker is the missing piece. It is NOT retry:\n"
        "    retry    = 'this one call failed, try it again'      (per call)\n"
        "    breaker  = 'this provider is down, stop calling it'  (across calls)\n"
        "\n  What it would need, and where it would go:\n"
        "    - a failure-rate window shared across calls, keyed by provider\n"
        "      (base_url + model), living beside RetryPolicy in agent/retry.py;\n"
        "    - open  -> fail fast with TransientLLMError, no HTTP at all, so\n"
        "               ReActLoop's llm_unavailable degradation fires in\n"
        "               milliseconds instead of minutes;\n"
        "    - half-open -> let exactly one probe through after a cool-down;\n"
        "    - closed -> normal operation.\n"
        "  Only *transient* failures may trip it: a PermanentLLMError says the\n"
        "  request was wrong, not the provider, and counting those would open\n"
        "  the circuit on a bug in our own prompt construction."
    )


def main() -> None:
    # The retry path logs llm.call.retrying / .rejected / .exhausted at
    # WARNING and above, which is the point in production and pure noise
    # here -- this demo narrates the same events in its own output. Raise
    # the threshold rather than removing the logging being demonstrated.
    logging.getLogger("agent").setLevel(logging.CRITICAL)

    print("Offline resilience demo -- no request leaves this process.")
    print("  every provider is a local fake; the one real OpenAILLM has its")
    print(f"  client replaced and points at {DISCARD_PORT_URL} (discard port)")
    demo_classification()
    demo_backoff()
    demo_total_deadline()
    demo_sdk_retries()
    demo_real_client_path()
    demo_loop_degradation()
    demo_load_shedding()
    demo_missing_circuit_breaker()
    print(
        "\nSummary: sections 1-7 are implemented and tested "
        "(tests/test_retry.py, tests/test_gateway.py).\n"
        "Section 8 is a measured gap, not a feature.\n"
    )


if __name__ == "__main__":
    main()
