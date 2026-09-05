"""A run that stops without ever producing a final answer must say so.

Budget-exceeded and fatal_tool_error stops already yielded an "error"
event, but running out of ``max_steps`` didn't, so the frontend trace just
stopped with nothing to show for it -- indistinguishable from a dropped
connection.

Since ``/api/stream`` was repointed at ``ReActLoop.aiter_run()`` (BUGS.md
#22) this is one code path, not a streaming-only one: the same
``run_completed`` event feeds the SSE error here and the ``stop_reason``
that ``POST /api/run`` returns.
"""

from __future__ import annotations

import asyncio

from app import server
from agent import LLMResponse, MockLLM, ToolCall, Usage


class _NeverFinishesLLM(MockLLM):
    """Never answers, and varies its arguments so it is not *looping*.

    The distinction matters: repeating one identical call is caught by loop
    detection, which is a different stop reason with a different message.
    This fixture exercises the plain "ran out of steps" path.
    """

    def __init__(self, **_):
        self.n = 0

    def chat(self, messages, tools=None):
        self.n += 1
        return LLMResponse(
            tool_calls=[
                ToolCall(id=f"c{self.n}", name="calculator",
                         arguments={"expression": f"{self.n}+1"})
            ],
            usage=Usage(1, 1),
        )


class _RepeatsOneCallLLM(MockLLM):
    """Always issues the identical call, which is what loop detection is for."""

    def chat(self, messages, tools=None):
        return LLMResponse(
            tool_calls=[ToolCall(id="c", name="datetime", arguments={})],
            usage=Usage(1, 1),
        )


async def _collect(task: str) -> str:
    response = await server.stream(task)
    return "".join([chunk async for chunk in response.body_iterator])


def test_max_steps_exhaustion_emits_a_visible_error_event(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NeverFinishesLLM)

    rendered = asyncio.run(_collect("keep going forever"))

    # The unified vocabulary: this is exactly the stop_reason POST /api/run
    # returns for the same run. The streaming path used to report a bare
    # "max_steps" of its own invention (BUGS.md #22).
    assert '"stop_reason": "budget: max_steps' in rendered
    assert '"success": false' in rendered
    assert "event: error" in rendered
    assert "Stopped after" in rendered
    assert "max_steps reached" in rendered
    # The error event must come before the terminal "done" event, not after
    # (a client rendering events as they arrive should see it before the
    # run visibly ends).
    assert rendered.index("event: error") < rendered.index('event: done')


def test_the_streaming_path_now_detects_a_loop_too(monkeypatch):
    """Loop detection is one of the things the hand-written streaming loop
    never had (BUGS.md #22): a model repeating one identical call ran to the
    full step budget instead of being cut off. Sharing ReActLoop means the
    streaming path inherits it rather than needing its own copy."""

    monkeypatch.setattr(server, "_build_llm", _RepeatsOneCallLLM)

    rendered = asyncio.run(_collect("keep going forever"))

    assert '"stop_reason": "loop_detected' in rendered
    assert "repeated call 'datetime'" in rendered
    assert "event: error" in rendered
    assert '"success": false' in rendered


def test_a_run_that_finishes_normally_gets_no_spurious_error_event(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)

    rendered = asyncio.run(_collect("What is 2 + 2?"))

    assert '"stop_reason": "finished"' in rendered
    assert "event: error" not in rendered
