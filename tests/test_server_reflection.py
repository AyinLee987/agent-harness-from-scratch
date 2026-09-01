"""A tool call failing used to let the model go straight from a failed
observation into another tool call, with nothing forcing it to pause and
reconsider -- the loop's own loop-same-call-limit guard only catches an
*identical* repeated (tool, arguments) pair, which a model varying its
arguments (or the URL/host it's hitting) each attempt never triggers. See
agent/trigger/react_loop.py's REFLECT_AFTER_FAILURE_STATE_KEY: a failed tool
call now forces the *next* think call to run with no tool schemas offered,
so the model must respond in plain text before it can act again.

app/server.py's _stream_leader_steps is a hand-rolled duplicate of
ReActLoop's think/act cycle for SSE streaming (the Leader's actual live path
via GET /api/stream) -- this file exercises that copy specifically, since a
fix in agent/trigger/react_loop.py alone would not reach it. See
tests/test_agent.py for the equivalent coverage of ReActLoop itself, which
subagents (and POST /api/run) always go through.
"""

from __future__ import annotations

import asyncio

from app import server
from agent import LLMResponse, MockLLM, ToolCall, Usage


class _FailsOnceThenAnswersLLM(MockLLM):
    """First real (tool-enabled) call requests an unknown tool; once forced
    into a tool-less reflection turn, it answers directly."""

    def __init__(self) -> None:
        super().__init__()
        self.real_calls = 0
        self.saw_tools_none = False

    def chat(self, messages, tools=None):
        if tools is None:
            self.saw_tools_none = True
            return LLMResponse(content="that failed; I'll answer directly.", usage=Usage(1, 1))
        self.real_calls += 1
        if self.real_calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="c", name="does_not_exist", arguments={})],
                usage=Usage(1, 1),
            )
        return LLMResponse(content="42", usage=Usage(1, 1))


async def _collect(task: str) -> str:
    response = await server.stream(task)
    return "".join([chunk async for chunk in response.body_iterator])


def test_a_failed_tool_call_forces_a_tool_less_reflection_turn(monkeypatch):
    llm = _FailsOnceThenAnswersLLM()
    monkeypatch.setattr(server, "_build_llm", lambda: llm)

    rendered = asyncio.run(_collect("call a tool that does not exist"))

    assert llm.saw_tools_none  # the forced turn really was offered no tools
    assert "event: reflection" in rendered
    assert "that failed" in rendered
    assert '"stop_reason": "finished"' in rendered
    assert '"text": "42"' in rendered
    # The reflection event must land between the failed tool_result and the
    # run's next think_start, not after the final answer.
    assert rendered.index("event: reflection") < rendered.index("event: answer")


def test_a_run_with_no_tool_failures_never_emits_a_reflection_event(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)

    rendered = asyncio.run(_collect("What is 2 + 2?"))

    assert "event: reflection" not in rendered
