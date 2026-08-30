"""Server-level checks for delegation as a normal Leader tool."""

from __future__ import annotations

import asyncio
import json

from app import server
from agent import LLMResponse, MockLLM, ToolCall, Usage


class _DelegatingLeaderLLM(MockLLM):
    def __init__(self) -> None:
        super().__init__()
        self.phase = 0

    def chat(self, messages, tools=None):
        tool_names = {item["function"]["name"] for item in tools or []}
        assert "spawn_subagent" in tool_names
        assert "calculator" in tool_names

        if self.phase == 0:
            self.phase = 1
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="spawn_subagent",
                        arguments={
                            "role": "analyst",
                            "task": "What is 2 + 2?",
                        },
                    )
                ],
                usage=Usage(1, 1),
            )
        if self.phase == 1:
            self.phase = 2
            spawn_result = next(
                json.loads(item["content"])
                for item in reversed(messages)
                if item["role"] == "tool" and item.get("name") == "spawn_subagent"
            )
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="wait",
                        name="wait_subagents",
                        arguments={
                            "task_ids": [spawn_result["task_id"]],
                            "timeout_seconds": 2.0,
                        },
                    )
                ],
                usage=Usage(1, 1),
            )
        return LLMResponse(content="Leader used the analyst result.", usage=Usage(1, 1))


def test_tools_endpoint_exposes_delegation_without_a_separate_mode(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)
    payload = asyncio.run(server.list_tools())
    names = {item["function"]["name"] for item in payload["tools"]}

    assert {"spawn_subagent", "wait_subagents", "calculator"} <= names
    assert not any(route.path == "/api/multi-agent/run" for route in server.app.routes)


def test_existing_stream_endpoint_can_delegate_and_reports_children(monkeypatch):
    leader_llm = _DelegatingLeaderLLM()
    calls = 0

    def build_llm():
        nonlocal calls
        calls += 1
        return leader_llm if calls == 1 else MockLLM()

    monkeypatch.setattr(server, "_build_llm", build_llm)

    async def collect():
        response = await server.stream("Use an analyst if useful")
        return "".join([chunk async for chunk in response.body_iterator])

    rendered = asyncio.run(collect())

    assert "spawn_subagent" in rendered
    assert "wait_subagents" in rendered
    assert '"subagents": [{' in rendered
    assert '"agent_name": "analyst"' in rendered
    assert '"success": true' in rendered

