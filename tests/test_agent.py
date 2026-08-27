"""Tests for the agent harness. All run against the dependency-free MockLLM."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"))

from agent import (
    ExecutionContext,
    FatalToolError,
    LongTermMemory,
    MockLLM,
    ReActAgent,
    RecoverableToolError,
    ShortTermMemory,
    ToolRegistry,
    tool,
)
from agent.eval.harness import EvalHarness
from basic_tools import build_agent, build_registry, calculator, web_search


# ---------------------------------------------------------------------------
# Tools / schema generation
# ---------------------------------------------------------------------------
def test_tool_decorator_builds_schema():
    schema = calculator.to_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "calculator"
    assert "expression" in fn["parameters"]["properties"]
    assert fn["parameters"]["properties"]["expression"]["type"] == "string"
    assert "expression" in fn["parameters"]["required"]
    # Docstring summary becomes the description.
    assert "arithmetic" in fn["description"].lower()


def test_tool_with_default_is_not_required():
    @tool
    def greet(name: str, excited: bool = False) -> str:
        """Greet someone.

        Args:
            name: Who to greet.
            excited: Whether to add an exclamation mark.
        """
        return f"Hi {name}{'!' if excited else ''}"

    params = greet.to_schema()["function"]["parameters"]
    assert "name" in params["required"]
    assert "excited" not in params["required"]
    assert params["properties"]["excited"]["type"] == "boolean"


def test_registry_dispatch_and_unknown():
    registry = build_registry()
    assert "calculator" in registry
    assert registry.dispatch("calculator", {"expression": "2 + 2"}) == "4"
    with pytest.raises(KeyError):
        registry.dispatch("nope", {})


def test_calculator_tool_safe_eval():
    assert calculator.run(expression="(12 + 8) * 5") == "100"
    # Malformed input is reported, not raised.
    assert "Could not evaluate" in calculator.run(expression="import os")


def test_web_search_stub():
    assert "Paris" in web_search.run(query="capital of France")
    assert "No results" in web_search.run(query="something obscure")


# ---------------------------------------------------------------------------
# ExecutionContext / budget guardrail
# ---------------------------------------------------------------------------
def test_context_budget_guard():
    ctx = ExecutionContext(max_steps=2, max_tokens=1000)
    assert not ctx.over_budget()
    ctx.new_step()
    ctx.new_step()
    assert ctx.over_budget()
    assert "max_steps" in ctx.budget_reason()

    ctx2 = ExecutionContext(max_steps=100, max_tokens=10)
    ctx2.add_tokens(50)
    assert ctx2.over_budget()
    assert "max_tokens" in ctx2.budget_reason()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
def test_agent_uses_tool_and_answers():
    agent = build_agent()
    result = agent.run("What is 23 times 17?")
    assert "391" in result.answer
    assert result.stop_reason == "finished"
    assert result.success
    # Trajectory recorded a calculator action.
    actions = [s["action"]["name"] for s in result.trajectory if s["action"]]
    assert "calculator" in actions


def test_agent_finishes_within_budget():
    agent = build_agent()
    result = agent.run("Search for the capital of France.")
    assert "Paris" in result.answer
    assert result.steps <= 6


def test_agent_handles_malformed_tool_call():
    """An LLM that requests a non-existent tool should be retried then fail cleanly."""

    from agent.llm import LLMResponse, ToolCall, Usage

    class BadLLM(MockLLM):
        def chat(self, messages, tools=None):
            # Always request a tool that does not exist.
            return LLMResponse(
                tool_calls=[ToolCall(id="x", name="ghost_tool", arguments={})],
                usage=Usage(1, 1),
            )

    agent = ReActAgent(llm=BadLLM(), tools=build_registry(), max_steps=4)
    result = agent.run("do something")
    # It should not crash; it stops via budget and returns a graceful message.
    assert result.answer
    assert "ERROR" in result.answer or "stopped" in result.answer.lower()


def test_recoverable_tool_error_is_returned_to_model():
    """A recoverable failure becomes a tool observation and the run continues."""

    from agent.llm import LLMResponse, ToolCall, Usage

    @tool
    def recoverable() -> str:
        raise RecoverableToolError("choose a different input")

    class RecoveryLLM(MockLLM):
        calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(id="r", name="recoverable", arguments={})],
                    usage=Usage(1, 1),
                )
            assert "choose a different input" in messages[-1]["content"]
            return LLMResponse(content="I handled the tool failure.", usage=Usage(1, 1))

    agent = ReActAgent(llm=RecoveryLLM(), tools=ToolRegistry([recoverable]))
    result = agent.run("recover")

    assert result.success
    assert result.stop_reason == "finished"
    assert "handled" in result.answer


def test_fatal_tool_error_aborts_run_without_model_follow_up():
    """A fatal failure stops the run and is not sent back for model recovery."""

    from agent.llm import LLMResponse, ToolCall, Usage

    @tool
    def fatal() -> str:
        raise FatalToolError("tool invariant was violated")

    class FatalLLM(MockLLM):
        calls = 0

        def chat(self, messages, tools=None):
            self.calls += 1
            return LLMResponse(
                tool_calls=[ToolCall(id="f", name="fatal", arguments={})],
                usage=Usage(1, 1),
            )

    llm = FatalLLM()
    result = ReActAgent(llm=llm, tools=ToolRegistry([fatal])).run("fail")

    assert llm.calls == 1
    assert not result.success
    assert result.stop_reason == "fatal_tool_error"
    assert "invariant" in result.answer
    assert result.trajectory[-1]["error"] == "tool invariant was violated"


def test_unclassified_tool_exception_is_fatal_by_default():
    """Python cannot enforce exhaustive mapping, so unknown failures fail closed."""

    from agent.llm import LLMResponse, ToolCall, Usage

    @tool
    def broken() -> str:
        raise RuntimeError("unexpected bug")

    class BrokenLLM(MockLLM):
        def chat(self, messages, tools=None):
            return LLMResponse(
                tool_calls=[ToolCall(id="b", name="broken", arguments={})],
                usage=Usage(1, 1),
            )

    result = ReActAgent(llm=BrokenLLM(), tools=ToolRegistry([broken])).run("fail")

    assert result.stop_reason == "fatal_tool_error"
    assert not result.success
    assert "unexpected bug" in result.answer


def test_decorator_can_classify_operation_errors_as_recoverable():
    @tool(error_policy="recoverable")
    def remote_operation() -> str:
        raise OSError("temporary network failure")

    with pytest.raises(RecoverableToolError, match="temporary network failure"):
        remote_operation.run()


def test_agent_detects_loop():
    """A model that repeats the same tool call is stopped as a loop, not just by the step budget."""

    from agent.llm import LLMResponse, ToolCall, Usage

    class LoopingLLM(MockLLM):
        def chat(self, messages, tools=None):
            # Never produces an answer — always re-requests the same call.
            return LLMResponse(
                tool_calls=[
                    ToolCall(id="x", name="calculator", arguments={"expression": "1 + 1"})
                ],
                usage=Usage(1, 1),
            )

    agent = ReActAgent(llm=LoopingLLM(), tools=build_registry(), max_steps=10)
    result = agent.run("keep going")
    assert result.stop_reason.startswith("loop_detected")
    assert result.steps < 10
    assert not result.success


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
def test_long_term_memory_recall():
    mem = LongTermMemory(MockLLM())
    mem.add("The project mascot is a red panda.")
    mem.add("The capital of France is Paris.")
    results = mem.search("Tell me about France", k=1)
    assert results
    assert "Paris" in results[0][0]


def test_short_term_memory_summarizes_when_over_budget():
    stm = ShortTermMemory(MockLLM(), window=2, max_tokens=5)
    messages = [{"role": "system", "content": "sys"}]
    for i in range(8):
        messages.append({"role": "user", "content": f"message number {i} with text"})
    managed = stm.manage(messages)
    # Should be compressed below the original length.
    assert len(managed) < len(messages)


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------
def test_eval_harness_scorecard():
    harness = EvalHarness(build_agent=build_agent)
    scorecard = harness.run()
    assert scorecard.total >= 10
    # The mock LLM should solve the large majority of deterministic tasks.
    assert scorecard.success_rate >= 0.8
    assert scorecard.avg_steps > 0
    rendered = scorecard.render()
    assert "EVAL SCORECARD" in rendered
