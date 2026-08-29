"""Leader/Worker orchestration tests using deterministic local LLMs."""

from __future__ import annotations

import json
import threading
import time

from agent import (
    AgentRegistry,
    AgentSpec,
    FatalToolError,
    LLMResponse,
    MockLLM,
    MultiAgentOrchestrator,
    ReActAgent,
    RunBudget,
    TaskStatus,
    ToolCall,
    ToolRegistry,
    Usage,
    tool,
)


class _ParallelTracker:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


class _WorkerLLM(MockLLM):
    def __init__(self, tracker: _ParallelTracker | None = None, delay: float = 0.0):
        super().__init__()
        self.tracker = tracker
        self.delay = delay

    def chat(self, messages, tools=None):
        if self.tracker is not None:
            self.tracker.enter()
        try:
            if self.delay:
                time.sleep(self.delay)
            task = next(m["content"] for m in messages if m["role"] == "user")
            return LLMResponse(content=f"worker completed: {task}", usage=Usage(1, 1))
        finally:
            if self.tracker is not None:
                self.tracker.leave()


class _DispatchTwoLeaderLLM(MockLLM):
    def __init__(self) -> None:
        super().__init__()
        self.phase = 0
        self.task_ids: list[str] = []

    def chat(self, messages, tools=None):
        if self.phase == 0:
            self.phase = 1
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="spawn-1",
                        name="spawn_subagent",
                        arguments={"role": "worker", "task": "first task"},
                    ),
                    ToolCall(
                        id="spawn-2",
                        name="spawn_subagent",
                        arguments={"role": "worker", "task": "second task"},
                    ),
                ],
                usage=Usage(1, 1),
            )
        if self.phase == 1:
            self.phase = 2
            spawned = [
                json.loads(m["content"])
                for m in messages
                if m["role"] == "tool" and m.get("name") == "spawn_subagent"
            ]
            self.task_ids = [item["task_id"] for item in spawned]
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="wait",
                        name="wait_subagents",
                        arguments={"task_ids": self.task_ids, "timeout_seconds": 2.0},
                    )
                ],
                usage=Usage(1, 1),
            )

        waited = next(
            m["content"]
            for m in reversed(messages)
            if m["role"] == "tool" and m.get("name") == "wait_subagents"
        )
        results = json.loads(waited)
        assert len(results) == 2
        assert all(item["success"] for item in results)
        return LLMResponse(content="leader combined both results", usage=Usage(1, 1))


def _leader(orchestrator: MultiAgentOrchestrator, llm: MockLLM) -> ReActAgent:
    return ReActAgent(llm=llm, tools=ToolRegistry(orchestrator.leader_tools()))


def test_leader_spawns_workers_in_parallel_and_waits_for_results():
    tracker = _ParallelTracker()
    registry = AgentRegistry()
    registry.register(
        AgentSpec("worker", "Completes one isolated unit of work."),
        lambda: ReActAgent(
            llm=_WorkerLLM(tracker, delay=0.15),
            tools=ToolRegistry(),
        ),
    )

    with MultiAgentOrchestrator(
        registry,
        RunBudget(max_parallel_tasks=2, subagent_timeout_seconds=2.0),
    ) as orchestrator:
        result = orchestrator.run_leader(
            _leader(orchestrator, _DispatchTwoLeaderLLM()),
            "delegate two tasks",
        )

    assert result.success
    assert result.answer == "leader combined both results"
    assert len(result.subagents) == 2
    assert all(item.success for item in result.subagents)
    assert tracker.peak == 2
    assert result.tokens >= 6


class _SingleDispatchLeaderLLM(MockLLM):
    def __init__(self, role: str = "worker", wait: bool = True) -> None:
        super().__init__()
        self.role = role
        self.wait = wait
        self.phase = 0

    def chat(self, messages, tools=None):
        if self.phase == 0:
            self.phase = 1
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="spawn_subagent",
                        arguments={"role": self.role, "task": "child task"},
                    )
                ],
                usage=Usage(1, 1),
            )
        spawn_output = next(
            m["content"]
            for m in reversed(messages)
            if m["role"] == "tool" and m.get("name") == "spawn_subagent"
        )
        if spawn_output.startswith("ERROR"):
            assert "Unknown subagent role" in spawn_output
            return LLMResponse(content="leader recovered", usage=Usage(1, 1))
        if not self.wait:
            return LLMResponse(content="leader finished early", usage=Usage(1, 1))
        if self.phase == 1:
            self.phase = 2
            task_id = json.loads(spawn_output)["task_id"]
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="wait",
                        name="wait_subagents",
                        arguments={"task_ids": [task_id], "timeout_seconds": 2.0},
                    )
                ],
                usage=Usage(1, 1),
            )
        child = json.loads(messages[-1]["content"])[0]
        assert not child["success"]
        assert child["stop_reason"] == "fatal_tool_error"
        return LLMResponse(content="leader handled child failure", usage=Usage(1, 1))


def test_child_fatal_error_is_data_for_leader_not_root_fatal():
    @tool
    def broken_worker_tool() -> str:
        raise FatalToolError("worker invariant failed")

    class FatalWorkerLLM(MockLLM):
        def chat(self, messages, tools=None):
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="fatal", name="broken_worker_tool", arguments={})
                ],
                usage=Usage(1, 1),
            )

    registry = AgentRegistry()
    registry.register(
        AgentSpec("worker", "A worker that fails fatally."),
        lambda: ReActAgent(
            llm=FatalWorkerLLM(),
            tools=ToolRegistry([broken_worker_tool]),
        ),
    )

    with MultiAgentOrchestrator(registry) as orchestrator:
        result = orchestrator.run_leader(
            _leader(orchestrator, _SingleDispatchLeaderLLM()),
            "delegate risky work",
        )

    assert result.success
    assert result.stop_reason == "finished"
    assert result.subagents[0].stop_reason == "fatal_tool_error"
    assert not result.subagents[0].success


def test_unknown_role_is_recoverable_for_leader():
    registry = AgentRegistry()
    with MultiAgentOrchestrator(registry) as orchestrator:
        result = orchestrator.run_leader(
            _leader(orchestrator, _SingleDispatchLeaderLLM(role="ghost")),
            "try an unavailable role",
        )

    assert result.success
    assert result.answer == "leader recovered"
    assert result.subagents == []


def test_orphan_worker_is_cancelled_when_leader_finishes():
    registry = AgentRegistry()
    registry.register(
        AgentSpec("worker", "A slow worker."),
        lambda: ReActAgent(
            llm=_WorkerLLM(delay=0.2),
            tools=ToolRegistry(),
        ),
    )

    with MultiAgentOrchestrator(registry) as orchestrator:
        result = orchestrator.run_leader(
            _leader(orchestrator, _SingleDispatchLeaderLLM(wait=False)),
            "finish without waiting",
        )

    assert result.success
    assert result.subagents[0].stop_reason == TaskStatus.CANCELLED.value
    assert not result.subagents[0].success


def test_agent_honors_a_preexisting_cancellation_request():
    cancelled = threading.Event()
    cancelled.set()
    llm = _WorkerLLM()
    result = ReActAgent(llm=llm, tools=ToolRegistry()).run(
        "do not start",
        cancellation_event=cancelled,
    )

    assert not result.success
    assert result.stop_reason == "cancelled"
    assert result.steps == 0

