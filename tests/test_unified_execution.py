"""Regression tests for BUGS.md #22 and #23.

``/api/stream`` used to be a hand-written second copy of the ReAct loop.
These pin the three symptoms that copy had, the properties that had to
survive replacing it, and the graph transition cap the unification
exposed.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace as _replace

import pytest

import app.server as server
from agent import (
    InMemoryCheckpointStore,
    JobBudget,
    JobRunner,
    MockLLM,
    ReActAgent,
    ToolRegistry,
    tool,
)
from agent.llm import LLMResponse, ToolCall, Usage
from agent.trigger.events import (
    RUN_COMPLETED,
    RUN_STARTED,
    TEXT,
    THINK_COMPLETED,
    THINK_STARTED,
    TOOL_COMPLETED,
    TOOL_STARTED,
)
from agent.trigger.react_loop import SUSPENDED_STOP_REASON


@tool
def slow(ms: int = 300) -> str:
    """A tool that blocks for a while."""
    time.sleep(ms / 1000.0)
    return "slow done"


@tool
def echo(text: str = "") -> str:
    """Echo text back."""
    return text


class _OneToolThenAnswer(MockLLM):
    def __init__(self, tool_name: str = "slow", **_) -> None:
        self.n = 0
        self.tool_name = tool_name

    def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            return LLMResponse(
                content="calling",
                tool_calls=[ToolCall(id="t1", name=self.tool_name, arguments={})],
                usage=Usage(1, 1),
            )
        return LLMResponse(content="final answer", usage=Usage(1, 1))


# -- symptom 1: the event loop is no longer blocked by a tool ---------------
def test_a_slow_tool_does_not_stall_the_event_loop():
    """Measured before the fix: a 300ms tool delayed a coroutine due at 30ms
    to 342ms, because the async streaming generator called the synchronous
    dispatcher inline. Every other request on the process waited too."""

    agent = ReActAgent(
        llm=_OneToolThenAnswer(), tools=ToolRegistry([slow]), max_steps=4
    )
    woke_at = []

    async def scenario():
        started = time.perf_counter()

        async def heartbeat():
            await asyncio.sleep(0.03)
            woke_at.append((time.perf_counter() - started) * 1000)

        async def consume():
            async for _event in agent.aiter_run("go"):
                pass

        await asyncio.gather(consume(), heartbeat())

    asyncio.run(scenario())

    assert woke_at[0] < 150, f"the loop was blocked for {woke_at[0]:.0f}ms"


# -- symptom 3: steps and tokens are real ----------------------------------
def test_the_streaming_path_reports_the_steps_it_actually_took():
    """It reported 0. The streaming loop counted with a local ``step_idx``
    and never called ``ctx.new_step()``, but the final stats read
    ``len(ctx.steps)``."""

    rendered = _stream("What is 2 + 2?")
    done = _event_payload(rendered, "done")

    assert done["steps"] >= 1
    assert done["tokens"] > 0


def test_the_prompt_counts_toward_the_streamed_token_total():
    """Token accounting was ``estimate_tokens_simple(full_content)`` --
    output text only. A ~1000-token prompt was recorded as 5, which also
    meant /api/stream had no working token budget."""

    agent = ReActAgent(
        llm=_OneToolThenAnswer(tool_name="echo"),
        tools=ToolRegistry([echo]),
        max_steps=4,
    )
    result = agent.run("a" * 4000)

    # MockLLM-driven Usage is tiny; what matters is that the *streamed*
    # path's estimate includes the input side. Drive it through aiter_run
    # against a genuinely streaming provider.
    streamed = _run_async(_StreamingLLM(), "a" * 4000)
    assert streamed.tokens > 500, streamed.tokens
    assert result.stop_reason == "finished"


class _StreamingLLM(MockLLM):
    """A provider that really streams, so TEXT events are exercised."""

    async def astream(self, messages, tools=None):
        for token in ["he", "ll", "o"]:
            await asyncio.sleep(0)
            yield {"type": "text", "data": token}


def _run_async(llm, task: str):
    agent = ReActAgent(llm=llm, tools=ToolRegistry([echo]), max_steps=4)
    result = {}

    async def go():
        async for event in agent.aiter_run(task):
            if event.kind == RUN_COMPLETED:
                result["r"] = event.data["result"]

    asyncio.run(go())
    return result["r"]


def test_a_streaming_provider_emits_token_events():
    agent = ReActAgent(llm=_StreamingLLM(), tools=ToolRegistry([echo]), max_steps=2)
    tokens = []

    async def go():
        async for event in agent.aiter_run("hi"):
            if event.kind == TEXT:
                tokens.append(event.data["token"])

    asyncio.run(go())
    assert tokens == ["he", "ll", "o"]


def test_a_non_streaming_provider_emits_no_token_events():
    """``BaseLLM.astream``'s default just wraps ``chat()``; driving it would
    block the loop while pretending to stream, so the driver must not."""

    agent = ReActAgent(llm=MockLLM(), tools=ToolRegistry([echo]), max_steps=2)
    kinds = []

    async def go():
        async for event in agent.aiter_run("What is 2 + 2?"):
            kinds.append(event.kind)

    asyncio.run(go())
    assert TEXT not in kinds
    assert RUN_COMPLETED in kinds


# -- the properties that had to survive ------------------------------------
def test_run_is_exactly_a_drain_of_iter_run():
    agent = ReActAgent(
        llm=_OneToolThenAnswer(tool_name="echo"),
        tools=ToolRegistry([echo]),
        max_steps=4,
    )
    generator = agent.iter_run("go")
    kinds = []
    while True:
        try:
            kinds.append(next(generator).kind)
        except StopIteration as done:
            drained = done.value
            break

    direct = ReActAgent(
        llm=_OneToolThenAnswer(tool_name="echo"),
        tools=ToolRegistry([echo]),
        max_steps=4,
    ).run("go")

    assert (drained.answer, drained.steps, drained.stop_reason) == (
        direct.answer,
        direct.steps,
        direct.stop_reason,
    )
    assert kinds[0] == RUN_STARTED
    assert kinds[-1] == RUN_COMPLETED
    assert THINK_STARTED in kinds and TOOL_STARTED in kinds and TOOL_COMPLETED in kinds


def test_the_sync_and_async_drivers_produce_the_same_run():
    """The whole point: one state machine, two transports."""

    def build():
        return ReActAgent(
            llm=_OneToolThenAnswer(tool_name="echo"),
            tools=ToolRegistry([echo]),
            max_steps=4,
        )

    sync = build().run("go")
    streamed = _collect_async(build(), "go")

    assert (sync.answer, sync.steps, sync.stop_reason, sync.success) == (
        streamed.answer,
        streamed.steps,
        streamed.stop_reason,
        streamed.success,
    )
    assert [c["name"] for s in sync.trajectory for c in s["tool_calls"]] == [
        c["name"] for s in streamed.trajectory for c in s["tool_calls"]
    ]


def _collect_async(agent, task: str):
    out = {}

    async def go():
        async for event in agent.aiter_run(task):
            if event.kind == RUN_COMPLETED:
                out["r"] = event.data["result"]

    asyncio.run(go())
    return out["r"]


def test_events_pair_up_one_per_tool_call():
    """TOOL_STARTED/TOOL_COMPLETED mirror Step.tool_calls (BUGS.md #16), so
    a UI and the trajectory cannot disagree about what ran."""

    agent = ReActAgent(
        llm=_OneToolThenAnswer(tool_name="echo"),
        tools=ToolRegistry([echo]),
        max_steps=4,
    )
    started, completed = [], []
    generator = agent.iter_run("go")
    while True:
        try:
            event = next(generator)
        except StopIteration as done:
            result = done.value
            break
        if event.kind == TOOL_STARTED:
            started.append(event.data["id"])
        elif event.kind == TOOL_COMPLETED:
            completed.append(event.data["id"])

    assert started == completed
    assert started == [
        call["id"] for step in result.trajectory for call in step["tool_calls"]
    ]


# -- BUGS.md #23: the graph's safety net must sit above the real budget -----
def test_the_graph_transition_cap_never_fires_before_the_step_budget():
    """StateGraph's 200-transition default is a "this graph cannot
    terminate" net, but a step costs 2-3 transitions, so it tripped at ~66
    steps -- below the shipped max_steps of 100 -- and reported
    ``max_transitions (200)`` instead of the real reason."""

    class _NeverAnswers(MockLLM):
        def __init__(self, **_):
            self.n = 0

        def chat(self, messages, tools=None):
            self.n += 1
            return LLMResponse(
                tool_calls=[
                    ToolCall(id=f"c{self.n}", name="echo", arguments={"text": str(self.n)})
                ],
                usage=Usage(1, 1),
            )

    result = ReActAgent(
        llm=_NeverAnswers(), tools=ToolRegistry([echo]), max_steps=100
    ).run("forever")

    assert "max_transitions" not in result.stop_reason
    assert result.stop_reason == "budget: max_steps (100) reached"
    assert result.steps == 100


# -- symptom 2: suspension, end to end through the endpoint ----------------
@tool
def slow_analysis(topic: str, job_context=None) -> str:
    """Analyse a topic (slowly)."""
    for _ in range(400):
        if job_context is not None:
            job_context.raise_if_cancelled()
            job_context.heartbeat("working")
        time.sleep(0.01)
    return f"analysis of {topic}"


class _JobUsingLLM(MockLLM):
    def __init__(self) -> None:
        self.calls = 0
        self.job_id = ""

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="a", name="slow_analysis", arguments={"topic": "widgets"})
                ],
                usage=Usage(1, 1),
            )
        if self.calls == 2:
            for message in reversed(messages):
                if message.get("role") == "tool":
                    self.job_id = json.loads(message["content"])["job_id"]
                    break
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="b",
                        name="await_jobs",
                        arguments={"job_ids": self.job_id, "timeout_seconds": 1},
                    )
                ],
                usage=Usage(1, 1),
            )
        return LLMResponse(content="the analysis is done", usage=Usage(1, 1))


@pytest.fixture
def jobs_server(monkeypatch):
    runner = JobRunner(budget=JobBudget(max_duration_seconds=30.0))
    llm = _JobUsingLLM()
    monkeypatch.setattr(server, "JOB_RUNNER", runner)
    monkeypatch.setattr(server, "CHECKPOINT_STORE", InMemoryCheckpointStore())
    monkeypatch.setattr(server, "REGISTRY", ToolRegistry([slow_analysis]))
    monkeypatch.setattr(server, "_build_llm", lambda: llm)
    monkeypatch.setattr(server, "_build_fast_llm", lambda: llm)
    monkeypatch.setattr(
        server,
        "CONFIG",
        _replace(
            server.CONFIG,
            jobs=_replace(
                server.CONFIG.jobs, enabled=True, long_running=["slow_analysis"]
            ),
        ),
    )
    try:
        yield server, runner, llm
    finally:
        runner.close()


def test_a_streamed_run_that_suspends_says_so_and_saves_a_checkpoint(jobs_server):
    """Before this, ``SuspendRun`` propagated out of the SSE generator
    uncaught: the stream died mid-run, no ``suspended`` event was emitted,
    zero checkpoints were saved, and the jobs it was waiting on were
    stranded with no run id to resume them under."""

    srv, _runner, llm = jobs_server

    rendered = _stream("analyse widgets")

    assert "event: suspended" in rendered
    suspended = _event_payload(rendered, "suspended")
    assert suspended["pending_job_ids"] == [llm.job_id]
    run_id = suspended["run_id"]
    assert run_id

    # And the id is one the resume endpoint actually knows.
    stored = srv.CHECKPOINT_STORE.load(run_id)
    assert stored is not None
    assert stored.pending_job_ids == [llm.job_id]

    done = _event_payload(rendered, "done")
    assert done["stop_reason"] == SUSPENDED_STOP_REASON
    assert done["run_id"] == run_id


# -- helpers ----------------------------------------------------------------
def _stream(task: str) -> str:
    async def collect() -> str:
        response = await server.stream(task)
        return "".join([chunk async for chunk in response.body_iterator])

    return asyncio.run(collect())


def _event_payload(rendered: str, name: str) -> dict:
    for block in rendered.split("\n\n"):
        if block.startswith(f"event: {name}\n"):
            return json.loads(block.split("data: ", 1)[1])
    raise AssertionError(f"no {name!r} event in:\n{rendered}")
