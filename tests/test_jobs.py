"""Tests for long-running tool execution, suspension and resume."""

from __future__ import annotations

import threading
import time

import pytest

from agent import (
    InMemoryJobStore,
    JobBudget,
    JobRunner,
    JobStatus,
    LongRunningTool,
    MockLLM,
    ReActAgent,
    SQLiteJobStore,
    SuspendRun,
    ToolRegistry,
    create_job_tools,
    tool,
)
from agent.jobs.models import Job
from agent.jobs.runner import fingerprint
from agent.llm import LLMResponse, ToolCall, Usage
from agent.state.context import ExecutionContext
from agent.trigger.react_loop import SUSPENDED_STOP_REASON


@tool
def slow_report(topic: str) -> str:
    """Produce a report (slowly)."""
    time.sleep(0.2)
    return f"report on {topic}"


@tool
def heartbeating_job(rounds: int = 5, job_context=None) -> str:
    """A tool that reports progress and honours cancellation."""
    for index in range(rounds):
        if job_context is not None:
            job_context.raise_if_cancelled()
            job_context.heartbeat(f"round {index + 1}/{rounds}")
        time.sleep(0.05)
    return "done"


def _wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


# -- the core contract ------------------------------------------------------
def test_calling_a_long_running_tool_returns_immediately_with_a_handle():
    """The rule the whole package exists for: a tool's execution time must
    not appear on the agent's call stack."""

    with JobRunner() as runner:
        wrapped = LongRunningTool(slow_report, runner)

        started = time.monotonic()
        observation = wrapped.run(topic="widgets")
        elapsed = time.monotonic() - started

        assert elapsed < 0.1, f"the call blocked for {elapsed:.2f}s"
        assert "job_id" in observation

        job_id = __import__("json").loads(observation)["job_id"]
        assert _wait_for(lambda: runner.get(job_id).status is JobStatus.SUCCEEDED)
        assert runner.get(job_id).result == "report on widgets"


def test_the_wrapped_tools_schema_is_preserved():
    """A tool becomes long-running by how it is registered, not by being
    rewritten -- the model calls it with exactly the same arguments."""

    with JobRunner() as runner:
        wrapped = LongRunningTool(slow_report, runner)
        assert wrapped.parameters_schema() == slow_report.parameters_schema()
        assert wrapped.name == slow_report.name


# -- idempotency ------------------------------------------------------------
def test_resubmitting_the_same_work_reuses_the_running_job():
    """A retrying model must not be able to start the same half-hour of
    work twice."""

    with JobRunner() as runner:
        first = runner.submit(slow_report, {"topic": "widgets"})
        second = runner.submit(slow_report, {"topic": "widgets"})

        assert first.job_id == second.job_id


def test_different_arguments_get_different_jobs():
    with JobRunner() as runner:
        first = runner.submit(slow_report, {"topic": "widgets"})
        second = runner.submit(slow_report, {"topic": "gadgets"})

        assert first.job_id != second.job_id


def test_a_failed_job_is_not_reused():
    """Retrying after a failure is the one case where the model genuinely
    does want the work done again."""

    @tool(error_policy="recoverable")
    def always_fails() -> str:
        """Always fails."""
        raise RuntimeError("nope")

    with JobRunner() as runner:
        first = runner.submit(always_fails, {})
        assert _wait_for(lambda: runner.get(first.job_id).status is JobStatus.FAILED)

        second = runner.submit(always_fails, {})
        assert second.job_id != first.job_id


def test_the_fingerprint_ignores_argument_spelling():
    assert fingerprint("t", {"a": 1, "b": 2}) == fingerprint("t", {"b": 2, "a": 1})
    assert fingerprint("t", {"a": 1}) != fingerprint("u", {"a": 1})


# -- heartbeat and cancellation ---------------------------------------------
def test_a_tool_that_accepts_a_job_context_reports_progress():
    with JobRunner() as runner:
        job = runner.submit(heartbeating_job, {"rounds": 5})
        assert _wait_for(
            lambda: (runner.get(job.job_id).progress or "").startswith("round")
        )
        assert _wait_for(
            lambda: runner.get(job.job_id).status is JobStatus.SUCCEEDED
        )


def test_cancellation_actually_reaches_inside_the_tool():
    """The gap this closes: MultiAgentOrchestrator sets an Event the loop
    only checks between steps, so a task blocked inside a tool cannot be
    stopped. A JobContext-aware tool can be."""

    with JobRunner() as runner:
        job = runner.submit(heartbeating_job, {"rounds": 200})
        assert _wait_for(lambda: runner.get(job.job_id).status is JobStatus.RUNNING)

        runner.cancel(job.job_id)

        assert runner.get(job.job_id).status is JobStatus.CANCELLED
        # And the worker thread really stopped rather than running to term.
        assert _wait_for(lambda: runner.get(job.job_id).finished_at is not None)


def test_a_cancelled_job_is_not_reported_as_a_failure(caplog):
    """Regression guard: JobCancelled used to be a plain exception, so
    FunctionTool wrapped it into a FatalToolError on the way out -- the
    runner's own `except JobCancelled` never matched and a deliberate
    cancellation was logged as `job.failed`."""

    import logging as _logging

    with JobRunner() as runner:
        job = runner.submit(heartbeating_job, {"rounds": 200})
        assert _wait_for(lambda: runner.get(job.job_id).status is JobStatus.RUNNING)

        with caplog.at_level(_logging.WARNING, logger="agent.jobs.runner"):
            runner.cancel(job.job_id)
            assert _wait_for(
                lambda: runner.get(job.job_id).finished_at is not None
            )
            time.sleep(0.2)

        assert "job.failed" not in caplog.text


def test_a_hard_duration_ceiling_applies_even_without_heartbeats():
    """A tool that never reports progress is not stall-watched -- it would
    trip instantly -- so max_duration_seconds is what bounds it."""

    @tool
    def forever() -> str:
        """Runs for a long time without reporting anything."""
        time.sleep(3.0)
        return "never seen"

    budget = JobBudget(max_duration_seconds=0.2)
    with JobRunner(budget=budget) as runner:
        job = runner.submit(forever, {})
        assert _wait_for(
            lambda: runner.get(job.job_id).status is JobStatus.TIMED_OUT, timeout=3.0
        )
        assert "max_duration" in runner.get(job.job_id).error


def test_a_watchdog_verdict_is_not_overwritten_by_a_late_return():
    """A tool that returns right after being timed out must not be able to
    overwrite the timeout the caller was already told about."""

    budget = JobBudget(max_duration_seconds=0.1)
    with JobRunner(budget=budget) as runner:
        @tool
        def slow_then_returns() -> str:
            """Returns after the deadline has already passed."""
            time.sleep(0.4)
            return "too late"

        job = runner.submit(slow_then_returns, {})
        assert _wait_for(
            lambda: runner.get(job.job_id).status is JobStatus.TIMED_OUT, timeout=3.0
        )
        time.sleep(0.5)
        assert runner.get(job.job_id).status is JobStatus.TIMED_OUT
        assert runner.get(job.job_id).result is None


# -- await_jobs -------------------------------------------------------------
def test_await_jobs_returns_results_once_everything_is_terminal():
    with JobRunner() as runner:
        registry = ToolRegistry(create_job_tools(runner))
        job = runner.submit(slow_report, {"topic": "widgets"})

        observation = registry.dispatch(
            "await_jobs", {"job_ids": job.job_id, "timeout_seconds": 5}
        )

        assert "report on widgets" in observation


def test_await_jobs_suspends_rather_than_erroring_when_work_is_still_running():
    """"Not finished yet" is the expected case on this path, not a failure."""

    with JobRunner() as runner:
        registry = ToolRegistry(create_job_tools(runner))
        job = runner.submit(heartbeating_job, {"rounds": 200})

        with pytest.raises(SuspendRun) as excinfo:
            registry.dispatch(
                "await_jobs", {"job_ids": job.job_id, "timeout_seconds": 1}
            )

        assert excinfo.value.job_ids == [job.job_id]
        runner.cancel(job.job_id)


def test_await_jobs_rejects_an_unknown_id_as_a_recoverable_error():
    from agent import RecoverableToolError

    with JobRunner() as runner:
        registry = ToolRegistry(create_job_tools(runner))
        with pytest.raises(RecoverableToolError, match="Unknown job_id"):
            registry.dispatch("await_jobs", {"job_ids": "nope"})


# -- suspend / resume through the loop --------------------------------------
def test_a_run_suspends_with_a_resumable_checkpoint_and_finishes_on_resume():
    """The end-to-end shape: start slow work, suspend, resume later and
    answer -- without the run's state ever having lived only in a thread."""

    with JobRunner() as runner:
        registry = ToolRegistry(
            [LongRunningTool(heartbeating_job, runner), *create_job_tools(runner)]
        )

        class _JobDrivingLLM(MockLLM):
            calls = 0

            def chat(self, messages, tools=None):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall(
                                id="a",
                                name="heartbeating_job",
                                arguments={"rounds": 3},
                            )
                        ],
                        usage=Usage(1, 1),
                    )
                if self.calls == 2:
                    return LLMResponse(
                        tool_calls=[
                            ToolCall(
                                id="b",
                                name="await_jobs",
                                arguments={
                                    "job_ids": _job_id_holder[0],
                                    "timeout_seconds": 1,
                                },
                            )
                        ],
                        usage=Usage(1, 1),
                    )
                return LLMResponse(content="the job finished", usage=Usage(1, 1))

        _job_id_holder = ["pending"]
        llm = _JobDrivingLLM()

        # Kick off the job through the tool so the id is real.
        import json as _json

        handle = _json.loads(registry.dispatch("heartbeating_job", {"rounds": 200}))
        _job_id_holder[0] = handle["job_id"]

        agent = ReActAgent(llm=llm, tools=registry, max_steps=6)
        result = agent.run("do the slow thing")

        assert result.stop_reason == SUSPENDED_STOP_REASON
        assert result.suspended
        assert result.checkpoint is not None
        assert result.pending_job_ids

        # The job lands, and the run picks up exactly where it stopped.
        runner.cancel(_job_id_holder[0])
        resumed = agent.run("", resume_from=result.checkpoint)

        assert resumed.stop_reason == "finished"
        assert resumed.steps > result.steps
        assert resumed.tokens >= result.tokens


def test_a_checkpoint_round_trips_the_whole_run_state():
    ctx = ExecutionContext(max_steps=7, max_tokens=999)
    ctx.add_message("system", "sys")
    ctx.add_message("user", "task")
    ctx.add_tokens(12)
    ctx.state["retries::x"] = 2
    step = ctx.new_step()
    step.thought = "thinking"
    step.record_tool_call(
        id="c1", name="x", arguments={}, observation="observed", ok=True
    )

    restored = ExecutionContext.restore(ctx.checkpoint())

    assert restored.run_id == ctx.run_id
    assert restored.messages == ctx.messages
    assert restored.tokens_used == 12
    assert restored.max_steps == 7
    assert restored.state["retries::x"] == 2
    assert [s.to_dict() for s in restored.steps] == ctx.trajectory()


def test_a_checkpoint_drops_unserializable_scratch_rather_than_failing():
    """A run that cannot be snapshotted at all is strictly worse than one
    snapshotted without some tool's cache object."""

    ctx = ExecutionContext()
    ctx.state["fine"] = 1
    ctx.state["a_lock"] = threading.Lock()

    snapshot = ctx.checkpoint()

    assert snapshot["state"] == {"fine": 1}


# -- store ------------------------------------------------------------------
@pytest.mark.parametrize("store_factory", [InMemoryJobStore, lambda: SQLiteJobStore(":memory:")])
def test_stores_round_trip_a_job(store_factory):
    store = store_factory()
    job = Job(job_id="j1", tool_name="t", fingerprint="f", progress="halfway")
    store.put(job)

    loaded = store.get("j1")
    assert loaded is not None
    assert loaded.tool_name == "t"
    assert loaded.progress == "halfway"
    assert store.list_unfinished() and store.list_unfinished()[0].job_id == "j1"

    job.status = JobStatus.SUCCEEDED
    job.result = "ok"
    store.put(job)
    assert store.get("j1").result == "ok"
    assert store.list_unfinished() == []
    assert store.delete("j1") is True
    assert store.get("j1") is None


@pytest.mark.parametrize("store_factory", [InMemoryJobStore, lambda: SQLiteJobStore(":memory:")])
def test_stores_only_reuse_a_job_inside_its_ttl(store_factory):
    store = store_factory()
    job = Job(job_id="j1", tool_name="t", fingerprint="f")
    job.created_at = time.time() - 100
    store.put(job)

    assert store.find_reusable("f", ttl_seconds=200) is not None
    assert store.find_reusable("f", ttl_seconds=10) is None


def test_a_sqlite_store_survives_reopening_the_same_file(tmp_path):
    """The point of durability here: a restart must not lose track of work
    that is already in flight."""

    path = str(tmp_path / "jobs.sqlite")
    store = SQLiteJobStore(path)
    store.put(Job(job_id="j1", tool_name="t", fingerprint="f", status=JobStatus.RUNNING))
    store.close()

    reopened = SQLiteJobStore(path)
    try:
        assert [job.job_id for job in reopened.list_unfinished()] == ["j1"]
    finally:
        reopened.close()
