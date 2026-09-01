"""A run that exhausts its step budget without ever producing a final
answer used to leave the SSE stream silent -- budget-exceeded and
fatal_tool_error stops already yielded an "error" event, but running out
of max_steps (the for loop simply ending) didn't, so the frontend trace
just stopped with nothing to show for it, indistinguishable from a dropped
connection. See app/server.py's _stream_leader_steps for-else.
"""

from __future__ import annotations

import asyncio

from app import server
from agent import LLMResponse, MockLLM, ToolCall, Usage


class _NeverFinishesLLM(MockLLM):
    """Always requests the same harmless tool call, never a final answer."""

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

    assert '"stop_reason": "max_steps"' in rendered
    assert '"success": false' in rendered
    assert "event: error" in rendered
    assert "Stopped after" in rendered
    assert "max_steps reached" in rendered
    # The error event must come before the terminal "done" event, not after
    # (a client rendering events as they arrive should see it before the
    # run visibly ends).
    assert rendered.index("event: error") < rendered.index('event: done')


def test_a_run_that_finishes_normally_gets_no_spurious_error_event(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)

    rendered = asyncio.run(_collect("What is 2 + 2?"))

    assert '"stop_reason": "finished"' in rendered
    assert "event: error" not in rendered
