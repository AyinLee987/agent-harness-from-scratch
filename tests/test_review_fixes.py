"""Regression tests for the defects in BUGS.md #9-#21.

Each of these reproduced against the pre-fix code. They are grouped by the
bug they guard rather than by module, because several of the fixes span two
files (a store primitive plus the caller that needed it) and the pairing is
the point.
"""

from __future__ import annotations

import time
from typing import Dict, List, Mapping, Optional, Sequence

import pytest

from agent import (
    MockLLM,
    ReActAgent,
    ToolRegistry,
    tool,
)
from agent.errors import RecoverableToolError
from agent.eval.harness import EvalHarness
from agent.jobs.models import Job, JobStatus
from agent.jobs.runner import JobRunner, fingerprint
from agent.jobs.store import InMemoryJobStore, SQLiteJobStore
from agent.llm import LLMResponse, ToolCall, Usage, parse_tool_arguments
from agent.memory.models import MemoryKind, MemoryRecord, MemoryStatus, utc_now
from agent.memory.repository import InMemoryMemoryRepository, SQLiteMemoryRepository
from agent.multi_agent.models import SubagentResult
from agent.retry import RetryPolicy, TransientLLMError, call_with_retry
from agent.state.context import ExecutionContext, Step
from agent.tools import _json_schema
from agent.trigger.dispatch import ToolDispatcher
from agent.trigger.gateway import AgentGateway


@tool
def echo(text: str) -> str:
    """Echo text back."""
    return text


@tool
def always_fails(x: str = "") -> str:
    """Always raises a recoverable error."""
    raise RecoverableToolError("boom")


# -- #9: container types in the generated schema ----------------------------
def test_a_container_parameter_keeps_its_container_type():
    """``get_origin()`` was treated as "this is a Union", so every generic
    collapsed to its first type argument: list[int] was advertised as
    ``integer``. A model following that schema sends a bare int."""

    assert _json_schema(list[int]) == {"type": "array", "items": {"type": "integer"}}
    assert _json_schema(dict[str, int]) == {
        "type": "object",
        "additionalProperties": {"type": "integer"},
    }
    assert _json_schema(List[str])["type"] == "array"
    assert _json_schema(Sequence[float]) == {
        "type": "array",
        "items": {"type": "number"},
    }
    assert _json_schema(Mapping[str, str])["type"] == "object"


def test_optional_still_describes_the_wrapped_type():
    assert _json_schema(Optional[int]) == {"type": "integer"}
    assert _json_schema(Optional[list[str]]) == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_a_heterogeneous_tuple_is_not_described_as_a_typed_array():
    """tuple[int, str] as "array of int" would be a confident lie; an
    untyped array is the honest answer."""

    assert _json_schema(tuple[int, str]) == {"type": "array"}
    assert _json_schema(tuple[int, ...]) == {
        "type": "array",
        "items": {"type": "integer"},
    }


def test_a_tools_generated_schema_reflects_its_annotations():
    @tool
    def batch(names: list[str], counts: dict[str, int], limit: int = 5) -> str:
        """Process a batch.

        Args:
            names: Which names to process.
            counts: Per-name counts.
            limit: Maximum to process.
        """
        return "ok"

    schema = batch.parameters_schema()
    assert schema["properties"]["names"]["type"] == "array"
    assert schema["properties"]["names"]["items"] == {"type": "string"}
    assert schema["properties"]["counts"]["additionalProperties"] == {"type": "integer"}
    assert schema["properties"]["names"]["description"] == "Which names to process."
    assert schema["required"] == ["names", "counts"]


# -- #10: malformed tool-argument JSON ---------------------------------------
def test_unparseable_tool_arguments_are_not_flattened_to_an_empty_call():
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments(None) == {}
    assert parse_tool_arguments("   ") == {}
    # Truncated mid-string: must survive as text, not become {}.
    assert parse_tool_arguments('{"text": "target-A') == '{"text": "target-A'


def test_a_tool_with_all_defaults_is_not_run_on_malformed_arguments():
    """The failure this guards: every parameter having a default makes ``{}``
    a *valid* call, so a truncated argument list silently became a call the
    model never made -- running with defaults it never chose, and with
    nothing in the transcript telling it so."""

    ran: List[str] = []

    @tool
    def send(target: str = "default-target") -> str:
        """Send to a target."""
        ran.append(target)
        return f"sent to {target}"

    ctx = ExecutionContext()
    observation = ToolDispatcher(ToolRegistry([send])).dispatch(
        ctx, "send", parse_tool_arguments('{"target": "prod-')
    )

    assert ran == []
    assert observation.startswith("ERROR")
    assert "not valid JSON" in observation
    assert "Please retry" in observation


# -- #11: job fingerprint ---------------------------------------------------
def test_the_job_fingerprint_is_case_sensitive():
    """It was lowercased, copied from ``_detect_loop``'s canonicalization.
    That is fine for a heuristic and wrong for an idempotency key: the key
    decides whether to hand back an existing job's *result*."""

    assert fingerprint("t", {"text": "Alpha"}) != fingerprint("t", {"text": "alpha"})
    assert fingerprint("t", {"url": "https://x/A"}) != fingerprint(
        "t", {"url": "https://x/a"}
    )


def test_the_job_fingerprint_still_ignores_key_order():
    assert fingerprint("t", {"a": 1, "b": 2}) == fingerprint("t", {"b": 2, "a": 1})


def test_case_differing_submissions_are_two_jobs():
    @tool
    def upper(text: str) -> str:
        """Uppercase text."""
        return text.upper()

    with JobRunner() as runner:
        first = runner.submit(upper, {"text": "Alpha"})
        second = runner.submit(upper, {"text": "alpha"})
        assert first.job_id != second.job_id


# -- #12: job state transitions ---------------------------------------------
@pytest.fixture(params=["memory", "sqlite"])
def job_store(request, tmp_path):
    if request.param == "memory":
        return InMemoryJobStore()
    return SQLiteJobStore(str(tmp_path / "jobs.sqlite"))


def _running_job(store, job_id: str = "j1") -> Job:
    job = Job(job_id=job_id, tool_name="slow", fingerprint="f", run_id="r", arguments={})
    job.status = JobStatus.RUNNING
    store.put(job)
    return job


def test_a_heartbeat_cannot_resurrect_a_cancelled_job(job_store):
    """The heartbeat was a read / mutate / whole-row write. A cancel landing
    between the read and the write was overwritten by the stale snapshot,
    turning a job the caller had been told was CANCELLED back into RUNNING."""

    _running_job(job_store)
    runner = JobRunner(store=job_store)
    runner.cancel("j1")
    assert job_store.get("j1").status is JobStatus.CANCELLED

    assert runner._heartbeat("j1", "still working") is None
    assert job_store.get("j1").status is JobStatus.CANCELLED
    runner.close()


def test_a_heartbeat_updates_progress_while_the_job_runs(job_store):
    _running_job(job_store)
    assert job_store.heartbeat("j1", at=123.0, progress="halfway") is True

    job = job_store.get("j1")
    assert job.heartbeat_at == 123.0
    assert job.progress == "halfway"
    assert job.status is JobStatus.RUNNING


def test_put_if_not_terminal_refuses_to_overwrite_a_terminal_verdict(job_store):
    job = _running_job(job_store)
    job.status = JobStatus.CANCELLED
    assert job_store.put_if_not_terminal(job) is True

    revived = job_store.get("j1")
    revived.status = JobStatus.RUNNING
    assert job_store.put_if_not_terminal(revived) is False
    assert job_store.get("j1").status is JobStatus.CANCELLED


def test_put_if_not_terminal_inserts_a_job_that_does_not_exist_yet(job_store):
    job = Job(job_id="new", tool_name="t", fingerprint="f", run_id="r", arguments={})
    assert job_store.put_if_not_terminal(job) is True
    assert job_store.get("new") is not None


def test_a_job_cancelled_before_its_worker_starts_stays_cancelled(job_store):
    """_execute set RUNNING with a plain put(), so a cancel that landed
    between submit() and the worker thread picking the job up was undone."""

    _running_job(job_store)
    runner = JobRunner(store=job_store)
    runner.cancel("j1")

    runner._execute("j1", echo, {"text": "hi"})
    assert job_store.get("j1").status is JobStatus.CANCELLED
    runner.close()


# -- #13: memory recall write-back ------------------------------------------
@pytest.fixture(params=["memory", "sqlite"])
def memory_repo(request, tmp_path):
    if request.param == "memory":
        return InMemoryMemoryRepository()
    return SQLiteMemoryRepository(str(tmp_path / "memory.sqlite"))


def _record(record_id: str = "m1") -> MemoryRecord:
    now = utc_now()
    return MemoryRecord(
        id=record_id,
        namespace="n",
        subject_id="s",
        kind=MemoryKind.USER_FACT,
        content="secret",
        source_type="explicit",
        content_hash="h",
        created_at=now,
        updated_at=now,
        metadata={},
    )


def test_touching_a_deleted_memory_does_not_revive_it(memory_repo):
    """recall() stamped the access time with a full-record update, so a
    tombstone committed between its read and its write was undone and the
    deleted content became searchable again."""

    memory_repo.insert(_record())
    live = memory_repo.get("m1")
    live.status = MemoryStatus.TOMBSTONED
    memory_repo.update(live)

    assert memory_repo.touch_if_active("m1", accessed_at=utc_now()) is None
    assert memory_repo.get("m1").status is MemoryStatus.TOMBSTONED


def test_touching_an_active_memory_stamps_it(memory_repo):
    memory_repo.insert(_record())
    stamp = utc_now()

    touched = memory_repo.touch_if_active("m1", accessed_at=stamp)

    assert touched is not None
    assert touched.last_accessed_at == stamp
    assert memory_repo.get("m1").last_accessed_at == stamp


def test_touching_a_missing_memory_returns_none(memory_repo):
    assert memory_repo.touch_if_active("nope", accessed_at=utc_now()) is None


# -- #14: facade attributes are not copies ----------------------------------
def test_writing_a_facade_attribute_reaches_the_loop():
    """These were plain copies, so writing one changed nothing the loop read
    -- silently. AgentGateway's per-request max_steps override was the live
    case: it set the copy and the run used the original budget."""

    agent = ReActAgent(llm=MockLLM(), tools=ToolRegistry([echo]), max_steps=10)

    agent.max_steps = 3
    assert agent._loop.max_steps == 3
    assert agent.max_steps == 3

    agent._loop.max_steps = 7
    assert agent.max_steps == 7


def test_the_gateways_per_request_step_override_is_applied_and_restored():
    calls = {"n": 0}

    class _Countdown(MockLLM):
        def chat(self, messages, tools=None):
            calls["n"] += 1
            return LLMResponse(
                content="working",
                tool_calls=[
                    ToolCall(id=str(calls["n"]), name="echo", arguments={"text": "x"})
                ],
                usage=Usage(1, 1),
            )

    agent = ReActAgent(llm=_Countdown(), tools=ToolRegistry([echo]), max_steps=10)
    result = AgentGateway(rate_limit=100, max_concurrency=4).run(
        agent, "go", max_steps=1
    )

    assert result.steps == 1
    assert calls["n"] == 1
    assert agent.max_steps == 10  # restored afterwards


# -- #15: the total deadline is a real ceiling ------------------------------
class _Status(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


def test_each_attempt_is_capped_by_the_remaining_total_budget():
    """The deadline used to be compared only against the next *backoff*, so
    a long attempt could sail past it: with a 10s deadline, a 9s attempt +
    0.1s backoff + another 9s attempt took 18s and was accepted."""

    budgets: List[float] = []
    clock = {"t": 0.0}

    def burn_nine(timeout: float):
        budgets.append(round(timeout, 2))
        clock["t"] += min(9.0, timeout)
        raise _Status(503)

    policy = RetryPolicy(
        timeout_seconds=60.0,
        total_deadline_seconds=10.0,
        initial_backoff=0.1,
        jitter=0.0,
        max_attempts=5,
    )
    monotonic = lambda: clock["t"]  # noqa: E731

    import agent.retry as retry_module

    original = retry_module.time.monotonic
    retry_module.time.monotonic = monotonic
    try:
        with pytest.raises(TransientLLMError, match="total deadline"):
            call_with_retry(
                burn_nine,
                policy=policy,
                operation="chat",
                sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
            )
    finally:
        retry_module.time.monotonic = original

    # First attempt clamped from 60s down to the 10s total budget; the
    # second gets only what is left.
    assert budgets[0] == 10.0
    assert budgets[1] < 1.0
    assert clock["t"] <= 10.0


def test_an_attempt_gets_the_policy_timeout_when_no_total_deadline_is_set():
    seen: List[float] = []

    def ok(timeout: float) -> str:
        seen.append(timeout)
        return "ok"

    policy = RetryPolicy(timeout_seconds=42.0, total_deadline_seconds=None)
    assert call_with_retry(ok, policy=policy, operation="chat") == "ok"
    assert seen == [42.0]


# -- #16: one trajectory record per tool call -------------------------------
class _TwoToolsLLM(MockLLM):
    def __init__(self) -> None:
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        if self.n == 1:
            return LLMResponse(
                content="calling both",
                tool_calls=[
                    ToolCall(id="a", name="echo", arguments={"text": "hi"}),
                    ToolCall(id="b", name="always_fails", arguments={"x": "y"}),
                ],
                usage=Usage(1, 1),
            )
        return LLMResponse(content="final", usage=Usage(1, 1))


def _two_tool_run():
    return ReActAgent(
        llm=_TwoToolsLLM(),
        tools=ToolRegistry([echo, always_fails]),
        max_steps=6,
    ).run("go")


def test_every_tool_call_in_a_turn_is_recorded_separately():
    """A turn calling two tools recorded only the first. If the *second*
    failed, its error text was concatenated onto the first's success, so
    the step read as one clean call."""

    step = _two_tool_run().trajectory[0]

    assert [(c["name"], c["ok"]) for c in step["tool_calls"]] == [
        ("echo", True),
        ("always_fails", False),
    ]


def test_the_flattened_view_is_still_populated_for_existing_consumers():
    step = _two_tool_run().trajectory[0]

    assert step["action"]["name"] == "echo"
    assert "hi" in step["observation"]
    assert "ERROR" in step["observation"]


def test_a_workers_tool_summary_reports_both_calls():
    result = _two_tool_run()
    summary = SubagentResult(
        task_id="t",
        agent_name="w",
        success=True,
        answer=result.answer,
        stop_reason=result.stop_reason,
        error_type=None,
        trajectory=result.trajectory,
    ).tool_call_summary()

    assert summary == [
        {"tool": "echo", "ok": True},
        {"tool": "always_fails", "ok": False},
    ]


def test_a_pre_existing_trajectory_without_per_call_records_still_summarizes():
    """Old checkpoints and stored runs have only the flattened view."""

    legacy = [
        {
            "index": 0,
            "action": {"name": "fetch", "arguments": {}},
            "observation": "ERROR calling 'fetch': blocked",
            "error": None,
        }
    ]
    summary = SubagentResult(
        task_id="t",
        agent_name="w",
        success=False,
        answer="",
        stop_reason="finished",
        error_type=None,
        trajectory=legacy,
    ).tool_call_summary()

    assert summary == [{"tool": "fetch", "ok": False}]


def test_a_legacy_step_backfills_one_call_record_on_restore():
    step = Step.from_dict(
        {
            "index": 0,
            "action": {"name": "fetch", "arguments": {"url": "u"}},
            "observation": "page text",
        }
    )

    assert [c["name"] for c in step.tool_calls] == ["fetch"]
    assert step.tool_calls[0]["ok"] is True


def test_the_tool_selector_never_hides_a_second_tool_from_the_same_turn():
    from agent.trigger.react_loop import _tools_used

    ctx = ExecutionContext()
    step = ctx.new_step()
    step.record_tool_call(id="a", name="echo", arguments={}, observation="hi", ok=True)
    step.record_tool_call(
        id="b", name="always_fails", arguments={}, observation="ERROR x", ok=False
    )

    assert _tools_used(ctx) == ["echo", "always_fails"]


# -- #17 / #18: eval scoring ------------------------------------------------
def _outcome(answer: str, tools: List[str], stop_reason: str = "finished"):
    from agent.agent import AgentResult

    trajectory = [
        {
            "index": i,
            "action": {"name": name, "arguments": {}},
            "observation": "ok",
            "error": None,
            "tool_calls": [
                {"id": str(i), "name": name, "arguments": {}, "observation": "ok", "ok": True}
            ],
        }
        for i, name in enumerate(tools)
    ]
    return AgentResult(
        answer=answer,
        success=stop_reason == "finished",
        steps=len(trajectory),
        tokens=1,
        stop_reason=stop_reason,
        trajectory=trajectory,
    )


def test_a_task_that_skipped_its_expected_tool_does_not_pass():
    """expect_tool was computed and then dropped from rule_pass, so an
    answer that never called the tool the task exists to exercise scored
    identically to one that did."""

    task = {"prompt": "p", "expect_substrings": ["42"], "expect_tool": "calculator"}

    assert EvalHarness._rule_score(task, _outcome("the answer is 42", [])) is False
    assert (
        EvalHarness._rule_score(task, _outcome("the answer is 42", ["calculator"]))
        is True
    )


def test_a_task_without_an_expected_tool_is_unaffected():
    task = {"prompt": "p", "expect_substrings": ["42"]}
    assert EvalHarness._rule_score(task, _outcome("42", [])) is True


def test_the_expected_tool_is_found_in_a_multi_call_turn():
    task = {"prompt": "p", "expect_substrings": [], "expect_tool": "second"}
    outcome = _outcome("x", [])
    outcome.trajectory = [
        {
            "index": 0,
            "action": {"name": "first", "arguments": {}},
            "observation": "ok",
            "tool_calls": [
                {"id": "a", "name": "first", "observation": "ok", "ok": True},
                {"id": "b", "name": "second", "observation": "ok", "ok": True},
            ],
        }
    ]

    assert EvalHarness._used_expected_tool(task, outcome) is True


class _JudgeLLM(MockLLM):
    def __init__(self, reply: str) -> None:
        self.reply = reply

    def chat(self, messages, tools=None):
        return LLMResponse(content=self.reply, usage=Usage(1, 1))


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("PASS", True),
        ("pass", True),
        ("FAIL", False),
        ("fail", False),
        # Substring matching on "pass" read every one of these as a pass.
        ("I cannot say whether this would pass.", False),
        ("This does not pass the bar.", False),
        ("", False),
        ("MAYBE", False),
    ],
)
def test_the_judge_reads_a_verdict_not_a_substring(reply, expected):
    harness = EvalHarness(build_agent=lambda: None, judge_llm=_JudgeLLM(reply))
    assert harness._judge_score({"prompt": "p"}, _outcome("a", [])) is expected


# -- #20: RAG version publishing --------------------------------------------
def _rag_service(repo):
    from agent.rag.chunking import MedicalParentChildChunker
    from agent.rag.ingestion import RAGIngestionService

    class _NullIndexer:
        def index_document(self, *args, **kwargs) -> None:
            return None

    return RAGIngestionService(repo, MedicalParentChildChunker(), [_NullIndexer()])


_BODY_A = "alpha 内容正文，足够长以便通过分块校验。" * 5
_BODY_B = "beta 内容正文，足够长以便通过分块校验。" * 5
_COMMON = dict(
    logical_id="doc-1",
    title="T",
    source_url="u",
    publisher="p",
    document_type="guideline",
    jurisdiction="CN",
    language="zh",
)


@pytest.fixture(params=["memory", "sqlite"])
def rag_repo(request, tmp_path):
    from agent.rag.repository import InMemoryRAGRepository, SQLiteRAGRepository

    if request.param == "memory":
        return InMemoryRAGRepository()
    return SQLiteRAGRepository(str(tmp_path / "rag.sqlite"))


def test_republishing_a_superseded_body_as_a_new_version_is_not_skipped(rag_repo):
    """Deduping on content alone meant a rollback silently did nothing:
    v1=alpha, v2=beta, then v3=alpha returned the *superseded* v1 with
    skipped=True while v2 stayed the effective version."""

    from agent.rag.models import DocumentStatus

    service = _rag_service(rag_repo)
    service.ingest_text(content=_BODY_A, version="1", **_COMMON)
    service.ingest_text(content=_BODY_B, version="2", **_COMMON)

    third = service.ingest_text(content=_BODY_A, version="3", **_COMMON)

    assert third.skipped is False
    assert third.document.version == "3"
    assert third.document.status is DocumentStatus.ACTIVE
    assert any("byte-identical" in w for w in third.warnings)

    active = [
        d for d in rag_repo.list_documents(status=DocumentStatus.ACTIVE)
        if d.logical_id == "doc-1"
    ]
    assert [d.version for d in active] == ["3"]


def test_reingesting_the_live_version_is_still_skipped(rag_repo):
    service = _rag_service(rag_repo)
    first = service.ingest_text(content=_BODY_A, version="1", **_COMMON)

    again = service.ingest_text(content=_BODY_A, version="1", **_COMMON)

    assert again.skipped is True
    assert again.document.id == first.document.id


# -- #21: the token budget accounts for the prompt --------------------------
def test_a_call_whose_prompt_alone_blows_the_budget_is_not_made():
    """over_budget() only saw tokens already spent, so a run one token under
    its ceiling could still issue a 40k-token call: max_tokens=1 finished
    successfully having spent 101."""

    calls = {"n": 0}

    class _Fat(MockLLM):
        def chat(self, messages, tools=None):
            calls["n"] += 1
            return LLMResponse(content="answer", usage=Usage(100, 1))

    result = ReActAgent(
        llm=_Fat(), tools=ToolRegistry([echo]), max_tokens=1
    ).run("go")

    assert calls["n"] == 0
    assert result.success is False
    assert "max_tokens" in result.stop_reason


def test_a_generous_budget_still_runs_normally():
    result = ReActAgent(
        llm=MockLLM(), tools=ToolRegistry([echo]), max_tokens=100_000
    ).run("say hello")

    assert result.stop_reason == "finished"


def test_the_offered_tool_schemas_count_toward_the_projected_prompt():
    from agent.trigger.react_loop import _prompt_tokens

    messages = [{"role": "user", "content": "hi"}]
    without = _prompt_tokens(messages, None)
    with_tools = _prompt_tokens(messages, ToolRegistry([echo, always_fails]).schemas())

    assert with_tools > without
