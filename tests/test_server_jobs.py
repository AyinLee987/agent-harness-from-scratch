"""End-to-end: a request whose tool takes too long suspends and resumes."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace as _replace

import pytest
from fastapi import HTTPException

import app.server as server
from agent import (
    InMemoryCheckpointStore,
    JobBudget,
    JobRunner,
    JobStatus,
    MockLLM,
    ToolRegistry,
    tool,
)
from agent.llm import LLMResponse, ToolCall, Usage
from agent.trigger.react_loop import SUSPENDED_STOP_REASON


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
    """Calls the slow tool, then waits on it, then answers."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.job_id = ""

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="a", name="slow_analysis", arguments={"topic": "widgets"}
                    )
                ],
                usage=Usage(1, 1),
            )
        if self.calls == 2:
            # Pick the job id out of the tool observation the loop appended.
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
    """A server whose Leader has one long-running tool and the job tools."""

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


def test_a_long_tool_suspends_the_request_instead_of_holding_it_open(jobs_server):
    """The point of the whole mechanism: the HTTP request returns in
    milliseconds while the work keeps going."""

    srv, runner, llm = jobs_server

    started = time.monotonic()
    result = asyncio.run(srv.run(srv.RunRequest(task="analyse widgets")))
    elapsed = time.monotonic() - started

    assert result.stop_reason == SUSPENDED_STOP_REASON
    assert result.run_id
    assert result.pending_job_ids == [llm.job_id]
    # The await_jobs tool waits its (1s) inline timeout before suspending;
    # what matters is that it is nowhere near the tool's own 4s duration.
    assert elapsed < 3.0, f"the request was held for {elapsed:.2f}s"

    assert runner.get(llm.job_id).status is JobStatus.RUNNING


def test_the_suspended_run_can_be_inspected_then_resumed_to_an_answer(jobs_server):
    srv, runner, llm = jobs_server

    suspended = asyncio.run(srv.run(srv.RunRequest(task="analyse widgets")))
    run_id = suspended.run_id

    status = asyncio.run(srv.suspended_run(run_id))
    assert status["status"] == "suspended"
    assert status["ready"] is False
    assert status["pending_jobs"][0]["job_id"] == llm.job_id

    # The job lands.
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if runner.get(llm.job_id).status is JobStatus.SUCCEEDED:
            break
        time.sleep(0.05)
    assert runner.get(llm.job_id).status is JobStatus.SUCCEEDED
    assert asyncio.run(srv.suspended_run(run_id))["ready"] is True

    resumed = asyncio.run(srv.resume_run(run_id))

    assert resumed.stop_reason == "finished"
    assert "done" in resumed.answer
    # The transcript carried across the suspension rather than restarting.
    assert resumed.steps > suspended.steps


def test_a_resumed_run_is_no_longer_resumable(jobs_server):
    """The checkpoint is deleted once the run reaches a terminal state, so a
    replayed resume request cannot run the same tail twice."""

    srv, runner, llm = jobs_server

    suspended = asyncio.run(srv.run(srv.RunRequest(task="analyse widgets")))
    runner.cancel(llm.job_id)
    asyncio.run(srv.resume_run(suspended.run_id))

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(srv.resume_run(suspended.run_id))
    assert excinfo.value.status_code == 404


def test_the_job_endpoints_report_and_cancel(jobs_server):
    srv, runner, llm = jobs_server

    asyncio.run(srv.run(srv.RunRequest(task="analyse widgets")))

    payload = asyncio.run(srv.job_status(llm.job_id))
    assert payload["job_id"] == llm.job_id
    assert payload["status"] in ("pending", "running")

    cancelled = asyncio.run(srv.cancel_job(llm.job_id))
    assert cancelled["status"] == "cancelled"

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(srv.job_status("nope"))
    assert excinfo.value.status_code == 404


def test_the_job_endpoints_are_503_when_jobs_are_disabled(monkeypatch):
    monkeypatch.setattr(server, "JOB_RUNNER", None)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(server.job_status("anything"))
    assert excinfo.value.status_code == 503


def test_a_non_long_running_tool_is_untouched_by_the_wrapper(monkeypatch):
    """Only the tools named in jobs.long_running change behaviour."""

    runner = JobRunner()
    try:
        monkeypatch.setattr(server, "JOB_RUNNER", runner)
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

        @tool
        def quick(text: str) -> str:
            """Returns immediately."""
            return text.upper()

        registry = server._register_job_tools(ToolRegistry([quick, slow_analysis]))

        assert registry.dispatch("quick", {"text": "hi"}) == "HI"
        assert "job_id" in registry.dispatch("slow_analysis", {"topic": "x"})
        assert "await_jobs" in registry.names()
    finally:
        runner.close()
