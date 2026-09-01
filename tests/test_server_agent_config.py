"""server._build_leader_runtime picks up config/agent.yaml (server.CONFIG)
instead of the old hardcoded max_steps=10 / RunBudget(...) literals.
"""

from __future__ import annotations

import asyncio

from app import server
from app.config import AgentConfig, LeaderConfig, ReActLoopConfig, RunBudgetConfig, WorkerConfig
from agent import LLMResponse, MockLLM, Usage


def _patched_config(**overrides) -> AgentConfig:
    base = AgentConfig()
    return AgentConfig(
        leader=overrides.get("leader", base.leader),
        worker=overrides.get("worker", base.worker),
        run_budget=overrides.get("run_budget", base.run_budget),
        react_loop=overrides.get("react_loop", base.react_loop),
        session=overrides.get("session", base.session),
    )


def test_leader_and_worker_use_independent_configured_step_budgets(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)
    monkeypatch.setattr(
        server,
        "CONFIG",
        _patched_config(
            leader=LeaderConfig(max_steps=17, max_tokens=12345),
            worker=WorkerConfig(max_steps=3),
        ),
    )

    orchestrator, leader = server._build_leader_runtime()
    try:
        assert leader.max_steps == 17
        assert leader.max_tokens == 12345
        researcher = orchestrator.registry.create("researcher")
        analyst = orchestrator.registry.create("analyst")
        assert researcher.max_steps == 3
        assert analyst.max_steps == 3
    finally:
        orchestrator.close()


def test_explicit_max_steps_overrides_both_leader_and_worker(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)
    monkeypatch.setattr(
        server,
        "CONFIG",
        _patched_config(leader=LeaderConfig(max_steps=17), worker=WorkerConfig(max_steps=3)),
    )

    orchestrator, leader = server._build_leader_runtime(max_steps=9)
    try:
        assert leader.max_steps == 9
        assert orchestrator.registry.create("researcher").max_steps == 9
        assert orchestrator.registry.create("analyst").max_steps == 9
    finally:
        orchestrator.close()


def test_run_budget_is_read_from_config(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)
    monkeypatch.setattr(
        server,
        "CONFIG",
        _patched_config(
            run_budget=RunBudgetConfig(
                max_subagents=2,
                max_parallel_tasks=1,
                max_depth=1,
                max_repeated_task=1,
                subagent_timeout_seconds=5.0,
            )
        ),
    )

    orchestrator, _leader = server._build_leader_runtime()
    try:
        assert orchestrator.budget.max_subagents == 2
        assert orchestrator.budget.max_parallel_tasks == 1
        assert orchestrator.budget.subagent_timeout_seconds == 5.0
    finally:
        orchestrator.close()


def test_react_loop_knobs_are_read_from_config(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)
    monkeypatch.setattr(
        server,
        "CONFIG",
        _patched_config(
            react_loop=ReActLoopConfig(
                max_tool_retries=4, loop_same_call_limit=9, compress_at_fraction=0.25
            )
        ),
    )

    orchestrator, leader = server._build_leader_runtime()
    try:
        assert leader._loop.max_tool_retries == 4
        assert leader._loop.loop_same_call_limit == 9
        assert leader._loop.compress_at_fraction == 0.25
    finally:
        orchestrator.close()


class _NameRememberingLLM(MockLLM):
    def chat(self, messages, tools=None):
        contents = [str(m.get("content") or "") for m in messages]
        if any("Zhuoyang" in c for c in contents):
            return LLMResponse(content="Your name is Zhuoyang.", usage=Usage(1, 1))
        return LLMResponse(content="I don't know your name.", usage=Usage(1, 1))


def test_run_request_omitting_max_steps_uses_the_configured_leader_default(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)
    monkeypatch.setattr(server, "CONFIG", _patched_config(leader=LeaderConfig(max_steps=7)))

    result = asyncio.run(server.run(server.RunRequest(task="hi")))

    assert result.success
