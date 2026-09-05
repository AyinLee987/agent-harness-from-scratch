"""Tests for intent routing in front of the ReAct loop."""

from __future__ import annotations

import asyncio
import json

import pytest

from agent import MockLLM, Route, StaticRouter
from agent.llm import LLMResponse, Usage
from agent.trigger.router import (
    ESCALATION_SENTINEL,
    LLMQueryRouter,
    wants_escalation,
)


class _VerdictLLM(MockLLM):
    """Returns one canned classifier verdict, and records what it was asked."""

    def __init__(self, payload) -> None:
        super().__init__()
        self.payload = payload
        self.prompts = []

    def chat(self, messages, tools=None):
        self.prompts.append(messages)
        content = (
            self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        )
        return LLMResponse(content=content, usage=Usage(5, 5))


# -- classification ---------------------------------------------------------
def test_a_conversational_turn_routes_direct_and_skips_retrieval():
    llm = _VerdictLLM(
        {"route": "direct", "normalized": "你好", "reasoning": "寒暄"}
    )
    plan = LLMQueryRouter(llm).route("你好呀！")

    assert plan.route is Route.DIRECT
    assert plan.needs_retrieval is False
    assert plan.original_task == "你好呀！"


def test_a_domain_question_routes_to_retrieval():
    llm = _VerdictLLM(
        {
            "route": "retrieval",
            "normalized": "布洛芬在孕期的用药安全性如何？",
            "reasoning": "需要权威医学资料",
        }
    )
    plan = LLMQueryRouter(llm).route("那个…孕妇能吃布洛芬吗")

    assert plan.route is Route.RETRIEVAL
    assert plan.needs_retrieval is True
    assert plan.task == "布洛芬在孕期的用药安全性如何？"
    assert plan.rewritten


def test_a_tool_task_routes_to_react_without_forcing_retrieval():
    """`react` deliberately does *not* set needs_retrieval: the search tool
    stays registered, it just isn't fired before the model has decided it
    needs evidence."""

    llm = _VerdictLLM(
        {"route": "react", "normalized": "计算 23 乘以 17", "reasoning": "需要计算"}
    )
    plan = LLMQueryRouter(llm).route("23 times 17 是多少")

    assert plan.route is Route.REACT
    assert plan.needs_retrieval is False


def test_the_domain_hint_reaches_the_classifier():
    llm = _VerdictLLM({"route": "direct", "normalized": "hi", "reasoning": ""})
    LLMQueryRouter(llm, domain_hint="医疗健康：症状分诊").route("hi")

    system = llm.prompts[0][0]["content"]
    assert "医疗健康：症状分诊" in system


# -- fail-open --------------------------------------------------------------
def test_an_unparseable_verdict_falls_open_to_the_full_loop():
    plan = LLMQueryRouter(_VerdictLLM("not json at all")).route("do a thing")

    assert plan.route is Route.REACT
    assert plan.task == "do a thing"


def test_an_unknown_route_value_falls_open_to_the_full_loop():
    llm = _VerdictLLM({"route": "teleport", "normalized": "x", "reasoning": ""})
    assert LLMQueryRouter(llm).route("do a thing").route is Route.REACT


def test_a_router_outage_falls_open_rather_than_taking_the_run_down():
    class BrokenLLM(MockLLM):
        def chat(self, messages, tools=None):
            raise RuntimeError("classifier endpoint is down")

    plan = LLMQueryRouter(BrokenLLM()).route("do a thing")

    assert plan.route is Route.REACT
    assert plan.task == "do a thing"


def test_a_rewrite_that_lost_the_question_is_discarded():
    """A "normalization" that shrinks a real question to a fragment has
    dropped it, not cleaned it up -- keep what the user actually typed."""

    llm = _VerdictLLM(
        {"route": "retrieval", "normalized": "药", "reasoning": "..."}
    )
    task = "孕妇在孕早期能不能服用布洛芬，以及推荐剂量是多少？"
    plan = LLMQueryRouter(llm).route(task)

    assert plan.task == task
    assert not plan.rewritten


# -- escalation -------------------------------------------------------------
def test_the_escalation_sentinel_is_recognised_even_when_wrapped():
    assert wants_escalation(ESCALATION_SENTINEL)
    assert wants_escalation(f"{ESCALATION_SENTINEL}.")
    assert wants_escalation(f"I think {ESCALATION_SENTINEL} here")
    assert not wants_escalation("Paris is the capital of France.")
    assert not wants_escalation(None)


# -- StaticRouter -----------------------------------------------------------
def test_static_router_pins_a_route_without_a_model():
    plan = StaticRouter(Route.RETRIEVAL).route("anything")
    assert plan.route is Route.RETRIEVAL
    assert plan.needs_retrieval is True


# -- server wiring ----------------------------------------------------------
def test_routing_is_off_by_default_and_keeps_mandatory_retrieval(monkeypatch):
    """Turning the router off must mean *exactly* the pre-router behaviour,
    including the unconditional RAG injection -- not a quietly different one."""

    import app.server as server

    plan = server._plan_run("anything at all")

    assert server.CONFIG.router.enabled is False
    assert plan.route is Route.REACT
    assert plan.needs_retrieval is True
    assert plan.task == "anything at all"


def test_a_direct_route_answers_without_building_a_leader(monkeypatch):
    import app.server as server
    from dataclasses import replace as _replace

    monkeypatch.setattr(server, "_build_llm", lambda: MockLLM())
    monkeypatch.setattr(server, "_build_fast_llm", lambda: MockLLM())
    monkeypatch.setattr(
        server, "CONFIG", _replace(
            server.CONFIG, router=_replace(server.CONFIG.router, enabled=True)
        )
    )
    monkeypatch.setattr(
        server, "_plan_run", lambda task: server.RunPlan(
            route=Route.DIRECT,
            task=task,
            original_task=task,
            needs_retrieval=False,
            reasoning="test",
        )
    )

    def _explode(*args, **kwargs):
        raise AssertionError("a direct route must not build the Leader runtime")

    monkeypatch.setattr(server, "_build_leader_runtime", _explode)

    result = asyncio.run(server.run(server.RunRequest(task="你好")))

    assert result.answer
    assert result.steps <= server.CONFIG.router.direct_max_steps


def test_a_direct_route_that_needs_tools_escalates_to_the_full_loop(monkeypatch):
    """The escape hatch that makes acting on a route safe: a misroute costs
    one extra model call, not a confidently wrong answer."""

    import app.server as server
    from dataclasses import replace as _replace

    class _EscalatingLLM(MockLLM):
        def chat(self, messages, tools=None):
            system = messages[0]["content"] if messages else ""
            if ESCALATION_SENTINEL in system:
                return LLMResponse(content=ESCALATION_SENTINEL, usage=Usage(1, 1))
            return super().chat(messages, tools)

    monkeypatch.setattr(server, "_build_llm", _EscalatingLLM)
    monkeypatch.setattr(server, "_build_fast_llm", _EscalatingLLM)
    monkeypatch.setattr(
        server, "CONFIG", _replace(
            server.CONFIG, router=_replace(server.CONFIG.router, enabled=True)
        )
    )
    monkeypatch.setattr(
        server, "_plan_run", lambda task: server.RunPlan(
            route=Route.DIRECT,
            task=task,
            original_task=task,
            needs_retrieval=False,
            reasoning="test",
        )
    )

    result = asyncio.run(server.run(server.RunRequest(task="What is 23 times 17?")))

    assert ESCALATION_SENTINEL not in result.answer
    assert "391" in result.answer
