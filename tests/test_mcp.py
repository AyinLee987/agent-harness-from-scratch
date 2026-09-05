"""Tests for MCP tools exposed through the synchronous ToolRegistry."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from mcp import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import MockLLM, RecoverableToolError, ToolRegistry, tool
from agent.mcp import (
    MCPConnectionError,
    MCPManager,
    MCPManagerClosedError,
    MCPServerConfig,
    MCPToolCallError,
)
from agent.mcp.launch import uv_tool_command
from agent.mcp.tool import format_mcp_result


class FakeClient:
    def __init__(self, remote_tools):
        self.remote_tools = remote_tools
        self.calls = []
        self.entered = False
        self.closed = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc, tb):
        self.closed = True

    async def list_tools(self):
        return SimpleNamespace(tools=self.remote_tools)

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        self.calls.append((name, arguments, read_timeout_seconds))
        return types.CallToolResult(
            content=[types.TextContent(text=f"called {name}: {arguments['url']}")]
        )


def _fetch_descriptor():
    return types.Tool(
        name="fetch",
        description="Fetch a URL and return readable content.",
        inputSchema={
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    )


def test_stdio_config_validates_and_redacts_secrets():
    config = MCPServerConfig.stdio(
        name="fetch",
        command="uvx",
        args=["mcp-server-fetch"],
        env={"TOKEN": "super-secret"},
    )
    assert config.transport == "stdio"
    assert config.command == "uvx"
    assert "super-secret" not in repr(config)

    with pytest.raises(ValueError):
        MCPServerConfig.stdio(name="", command="uvx")
    with pytest.raises(ValueError):
        MCPServerConfig.http(name="remote", url="ftp://example.com/mcp")


def test_importing_base_agent_does_not_require_optional_mcp_dependency():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = """
import builtins
original_import = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "mcp" or name.startswith("mcp."):
        raise ModuleNotFoundError("blocked optional dependency", name="mcp")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked
import agent
print(agent.ReActAgent.__name__)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ReActAgent"


def test_manager_discovers_and_calls_namespaced_tool():
    client = FakeClient([_fetch_descriptor()])
    config = MCPServerConfig.stdio(
        name="fetch", command="uvx", args=["mcp-server-fetch"]
    )

    with MCPManager([config], client_factory=lambda _: client) as manager:
        tools = manager.tools()
        assert [t.name for t in tools] == ["fetch__fetch"]
        schema = tools[0].parameters_schema()
        assert schema["required"] == ["url"]
        assert tools[0].run(url="https://example.com") == (
            "called fetch: https://example.com"
        )

    assert client.entered
    assert client.closed
    assert client.calls == [
        ("fetch", {"url": "https://example.com"}, config.call_timeout)
    ]
    with pytest.raises(MCPManagerClosedError):
        manager.call("fetch", "fetch", {"url": "https://example.com"})


class SlowClient(FakeClient):
    """Every call takes ``delay`` seconds and records how many overlap."""

    def __init__(self, remote_tools, delay=0.3):
        super().__init__(remote_tools)
        self.delay = delay
        self.in_flight = 0
        self.peak_in_flight = 0

    async def call_tool(self, name, arguments, read_timeout_seconds=None):
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await asyncio.sleep(self.delay)
            return types.CallToolResult(
                content=[types.TextContent(text=f"called {name}: {arguments['url']}")]
            )
        finally:
            self.in_flight -= 1


def _call_from_threads(manager, count):
    """Fire ``count`` concurrent manager.call()s and return (results, seconds)."""

    results: list = [None] * count
    errors: list = [None] * count

    def worker(index):
        try:
            results[index] = manager.call(
                "fetch", "fetch", {"url": f"https://example.com/{index}"}
            )
        except Exception as exc:  # recorded so the assert can show it
            errors[index] = exc

    started = time.monotonic()
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    elapsed = time.monotonic() - started

    assert not any(errors), errors
    return results, elapsed


def test_concurrent_calls_overlap_instead_of_taking_turns():
    """Regression guard: the worker loop used to ``await`` each call inline,
    so N parallel Workers calling MCP tools were serialized behind one
    another -- N x the latency, and the ones still queued burned their
    caller-side timeout without ever having been sent."""

    client = SlowClient([_fetch_descriptor()], delay=0.3)
    config = MCPServerConfig.stdio(
        name="fetch", command="uvx", max_concurrent_calls=4
    )

    with MCPManager([config], client_factory=lambda _: client) as manager:
        results, elapsed = _call_from_threads(manager, 4)

    assert len(results) == 4
    assert all(result for result in results)
    assert client.peak_in_flight == 4
    # Serialized, four 0.3s calls would take >= 1.2s.
    assert elapsed < 0.9, f"calls appear serialized ({elapsed:.2f}s)"


def test_per_server_concurrency_is_bounded_by_max_concurrent_calls():
    """Unbounded overlap is not the fix -- the cap is an explicit guardrail
    the way max_steps/max_tokens are, so a Leader fanning out cannot flood
    whatever is on the other end of the transport."""

    client = SlowClient([_fetch_descriptor()], delay=0.2)
    config = MCPServerConfig.stdio(
        name="fetch", command="uvx", max_concurrent_calls=2
    )

    with MCPManager([config], client_factory=lambda _: client) as manager:
        results, _ = _call_from_threads(manager, 6)

    assert len(results) == 6
    assert client.peak_in_flight == 2


def test_a_call_that_never_gets_a_slot_says_so_rather_than_claiming_a_timeout():
    """A call still waiting its turn has not timed out -- it was never sent.
    Reporting the two cases identically is what made the original bug so
    hard to read from logs alone."""

    client = SlowClient([_fetch_descriptor()], delay=1.0)
    config = MCPServerConfig.stdio(
        name="fetch",
        command="uvx",
        max_concurrent_calls=1,
        slot_wait_timeout=0.05,
        call_timeout=5.0,
    )

    with MCPManager([config], client_factory=lambda _: client) as manager:
        blocker = threading.Thread(
            target=lambda: manager.call("fetch", "fetch", {"url": "https://a"})
        )
        blocker.start()
        time.sleep(0.1)
        try:
            with pytest.raises(MCPToolCallError, match="free slot"):
                manager.call("fetch", "fetch", {"url": "https://b"})
        finally:
            blocker.join(timeout=10)


def test_uv_tool_command_uses_native_binary_beside_python(tmp_path):
    python = tmp_path / "python.exe"
    uv = tmp_path / "uv.exe"
    python.touch()
    uv.touch()

    command, args = uv_tool_command(
        "mcp-server-fetch",
        python_executable=str(python),
        path_lookup=lambda _: None,
    )
    assert command == str(uv)
    assert args == ["tool", "run", "mcp-server-fetch"]


def test_nested_connection_errors_include_the_real_cause():
    class BrokenClient:
        async def __aenter__(self):
            raise ExceptionGroup(
                "stdio failed", [FileNotFoundError("uv executable missing")]
            )

        async def __aexit__(self, exc_type, exc, tb):
            return None

    config = MCPServerConfig.stdio(name="fetch", command="missing")
    with pytest.raises(
        MCPConnectionError,
        match=r"fetch.*FileNotFoundError.*uv executable missing",
    ):
        with MCPManager([config], client_factory=lambda _: BrokenClient()):
            pass


def test_result_conversion_handles_text_structured_media_and_errors():
    assert issubclass(MCPToolCallError, RecoverableToolError)
    result = types.CallToolResult(
        content=[
            types.TextContent(text="hello"),
            types.ImageContent(data="ZmFrZQ==", mimeType="image/png"),
        ],
        structuredContent={"status": "ok"},
    )
    rendered = format_mcp_result(result)
    assert "hello" in rendered
    assert "[image: image/png]" in rendered
    assert '"status": "ok"' in rendered

    with pytest.raises(MCPToolCallError):
        format_mcp_result(
            types.CallToolResult(
                content=[types.TextContent(text="permission denied")],
                isError=True,
            )
        )


def test_registry_rejects_duplicate_tool_names():
    @tool("same")
    def first() -> str:
        return "first"

    @tool("same")
    def second() -> str:
        return "second"

    registry = ToolRegistry([first])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(second)


def test_mock_llm_selects_fetch_mcp_for_explicit_fetch_request():
    llm = MockLLM()
    response = llm.chat(
        [{"role": "user", "content": "Fetch https://example.com"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search a canned knowledge base",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "fetch__fetch",
                    "description": "Fetch a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {"url": {"type": "string"}},
                        "required": ["url"],
                    },
                },
            },
        ],
    )
    assert response.tool_calls[0].name == "fetch__fetch"
    assert response.tool_calls[0].arguments == {"url": "https://example.com"}
