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
    assert '"status": "succeeded"' in rendered


class _FireAndForgetLeaderLLM(MockLLM):
    """Spawns a Worker and finishes immediately -- never calls
    wait_subagents or get_subagent_status. Regression fixture for the
    playground bug where a dispatched subagent's card stayed stuck on
    'pending' forever: the final "done" event's subagents list is the
    *only* update the client gets once the Leader stops polling on its own.
    """

    def chat(self, messages, tools=None):
        tool_names = {item["function"]["name"] for item in tools or []}
        if "spawn_subagent" in tool_names and not any(
            item.get("name") == "spawn_subagent"
            for item in messages
            if item["role"] == "tool"
        ):
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="spawn",
                        name="spawn_subagent",
                        arguments={"role": "analyst", "task": "What is 2 + 2?"},
                    )
                ],
                usage=Usage(1, 1),
            )
        return LLMResponse(content="Dispatched, moving on.", usage=Usage(1, 1))


def test_done_event_reports_a_final_status_even_when_the_leader_never_polls_again(monkeypatch):
    leader_llm = _FireAndForgetLeaderLLM()
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

    done_line = next(
        line for line in rendered.splitlines() if line.startswith("data: ") and '"subagents"' in line
    )
    done_payload = json.loads(done_line[len("data: ") :])
    assert len(done_payload["subagents"]) == 1
    # The Worker must have reached some terminal state (succeeded on its own,
    # or cancelled when the root closed behind it) by the time "done" fires
    # -- never left as "pending", which is what a client can't distinguish
    # from "still running" and would render as stuck forever.
    assert done_payload["subagents"][0]["status"] != "pending"

