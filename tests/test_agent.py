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
            if tools is None:
                # Forced reflection turn after the failure (see
                # REFLECT_AFTER_FAILURE_STATE_KEY) -- no tools offered, so
                # this call can only respond in plain text.
                assert any(
                    "choose a different input" in str(m.get("content", ""))
                    for m in messages
                )
                return LLMResponse(content="that failed; I'll try differently.", usage=Usage(1, 1))
            return LLMResponse(content="I handled the tool failure.", usage=Usage(1, 1))

    agent = ReActAgent(llm=RecoveryLLM(), tools=ToolRegistry([recoverable]))
    result = agent.run("recover")

    assert result.success
    assert result.stop_reason == "finished"
    assert "handled" in result.answer


def test_a_failed_tool_call_forces_a_tool_less_reflection_turn_before_the_next_action():
    """A model going straight from a failed observation into another tool
    call, with nothing pausing it to reconsider, used to be how a retry
    spiral happened -- loop detection only catches an *identical* repeated
    (tool, arguments) pair, not one that varies its arguments each attempt.
    Every failed tool call now forces the *next* think call to run with no
    tool schemas offered (REFLECT_AFTER_FAILURE_STATE_KEY in
    agent/trigger/react_loop.py), applying to both a Leader and any Worker
    it spawns since both are ReActAgent/ReActLoop underneath."""

    from agent import FORCED_REFLECTION_PROMPT
    from agent.llm import LLMResponse, ToolCall, Usage

    @tool
    def flaky() -> str:
        raise RecoverableToolError("temporarily unavailable")

    class ProbeLLM(MockLLM):
        def __init__(self) -> None:
            super().__init__()
            self.tools_offered_per_call: list[bool] = []
            self.reflection_call_messages: list[dict] | None = None

        def chat(self, messages, tools=None):
            self.tools_offered_per_call.append(tools is not None)
            if tools is None:
                self.reflection_call_messages = list(messages)
                return LLMResponse(content="reflecting on the failure", usage=Usage(1, 1))
            if len(self.tools_offered_per_call) == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(id="f", name="flaky", arguments={})],
                    usage=Usage(1, 1),
                )
            return LLMResponse(content="giving up after reflecting", usage=Usage(1, 1))

    llm = ProbeLLM()
    result = ReActAgent(llm=llm, tools=ToolRegistry([flaky])).run("try the flaky tool")

    # Call order: [tools offered] -> fails -> [no tools, forced] -> [tools offered] -> answers.
    assert llm.tools_offered_per_call == [True, False, True]
    assert result.success
    assert result.stop_reason == "finished"
    assert result.answer == "giving up after reflecting"

    # The reflection prompt that solicited the tool-less turn is really in
    # the transcript that turn saw.
    assert llm.reflection_call_messages is not None
    assert any(
        m.get("role") == "user" and m.get("content") == FORCED_REFLECTION_PROMPT
        for m in llm.reflection_call_messages
    )

    reflection_steps = [s for s in result.trajectory if s["observation"] == "reflection"]
    assert len(reflection_steps) == 1
    assert reflection_steps[0]["action"] is None
    assert reflection_steps[0]["thought"] == "reflecting on the failure"
    assert reflection_steps[0]["error"] is None


def test_repeated_failures_against_the_same_host_get_a_switch_source_hint():
    """A model retrying a blocked site with a different URL each time never
    repeats the identical (tool, arguments) pair loop detection watches for
    -- this is the narrower per-host signal that catches that case (see
    agent/trigger/dispatch.py's _source_streak_key docstring)."""

    from agent.llm import LLMResponse, ToolCall, Usage

    @tool
    def fetch_like(url: str) -> str:
        raise RecoverableToolError("403 Forbidden")

    class HammeringLLM(MockLLM):
        def __init__(self) -> None:
            super().__init__()
            self.real_calls = 0

        def chat(self, messages, tools=None):
            if tools is None:
                # Forced reflection turn after a failure (see
                # REFLECT_AFTER_FAILURE_STATE_KEY) -- doesn't count as one
                # of the model's real tool-calling decisions below.
                return LLMResponse(content="that failed, trying another page", usage=Usage(1, 1))
            self.real_calls += 1
            if self.real_calls <= 3:
                # A different path each time -- never the identical call.
                return LLMResponse(
                    tool_calls=[ToolCall(
                        id=str(self.real_calls), name="fetch_like",
                        arguments={"url": f"https://blocked.example/page{self.real_calls}"},
                    )],
                    usage=Usage(1, 1),
                )
            return LLMResponse(content="giving up on that host", usage=Usage(1, 1))

    agent = ReActAgent(llm=HammeringLLM(), tools=ToolRegistry([fetch_like]))
    result = agent.run("look something up")

    assert result.success
    assert result.stop_reason == "finished"  # loop detection never fired
    observations = [step["observation"] for step in result.trajectory if step.get("action")]
    assert len(observations) == 3
    # First failure: no hint yet (threshold is 2).
    assert "Try a different source" not in observations[0]
    # Second and third consecutive failures against the same host: hinted.
    assert "Try a different source" in observations[1]
    assert "Try a different source" in observations[2]


def test_source_failure_streak_resets_on_success_and_is_per_host():
    from agent.llm import LLMResponse, ToolCall, Usage

    class FlakyThenOkLLM(MockLLM):
        def __init__(self) -> None:
            super().__init__()
            self.real_calls = 0

        def chat(self, messages, tools=None):
            if tools is None:
                # Forced reflection turn after the first failure.
                return LLMResponse(content="trying a different host", usage=Usage(1, 1))
            self.real_calls += 1
            if self.real_calls == 1:
                return LLMResponse(
                    tool_calls=[ToolCall(
                        id="a", name="fetch_like",
                        arguments={"url": "https://blocked.example/a"},
                    )],
                    usage=Usage(1, 1),
                )
            if self.real_calls == 2:
                # Different host -- must not inherit blocked.example's streak.
                return LLMResponse(
                    tool_calls=[ToolCall(
                        id="b", name="fetch_like",
                        arguments={"url": "https://ok.example/b"},
                    )],
                    usage=Usage(1, 1),
                )
            return LLMResponse(content="done", usage=Usage(1, 1))

    @tool
    def fetch_like(url: str) -> str:
        if "blocked.example" in url:
            raise RecoverableToolError("403 Forbidden")
        return "page content"

    agent = ReActAgent(llm=FlakyThenOkLLM(), tools=ToolRegistry([fetch_like]))
    result = agent.run("look something up")

    assert result.success
    observations = [step["observation"] for step in result.trajectory if step.get("action")]
    assert "Try a different source" not in observations[0]  # first failure, below threshold
    assert observations[1] == "page content"  # different host, unaffected by the streak


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


def test_short_term_memory_never_separates_a_tool_calls_message_from_its_tool_responses():
    # A naive tail-slice window can cut between an assistant's tool_calls
    # message and the tool responses that answer it — e.g. exactly where a
    # multi-agent delegation step (assistant + 2 parallel spawn_subagent
    # calls + 2 tool results) happens to fall relative to the window
    # boundary. Every OpenAI-compatible chat API rejects a 'tool' message
    # whose triggering tool_calls message isn't present, so that shape is a
    # 400 waiting to happen, not just a lossy summary.
    stm = ShortTermMemory(MockLLM(), window=12, max_tokens=4000)
    messages = [{"role": "system", "content": "sys"}]
    for i in range(5):
        messages.append({"role": "user", "content": f"turn {i}"})
        messages.append({"role": "assistant", "content": f"reply {i}"})
    messages.append({
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "call_A", "type": "function", "function": {"name": "spawn_subagent", "arguments": "{}"}},
            {"id": "call_B", "type": "function", "function": {"name": "spawn_subagent", "arguments": "{}"}},
        ],
    })
    messages.append({"role": "tool", "content": "result A", "tool_call_id": "call_A", "name": "spawn_subagent"})
    messages.append({"role": "tool", "content": "result B", "tool_call_id": "call_B", "name": "spawn_subagent"})
    for i in range(5, 10):
        messages.append({"role": "user", "content": f"turn {i}"})
        messages.append({"role": "assistant", "content": f"reply {i}"})

    managed = stm.manage(messages)

    seen_ids: set = set()
    for m in managed:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            seen_ids = {tc["id"] for tc in m["tool_calls"]}
        elif m.get("role") == "tool":
            assert m.get("tool_call_id") in seen_ids, f"orphaned tool message: {m}"


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
