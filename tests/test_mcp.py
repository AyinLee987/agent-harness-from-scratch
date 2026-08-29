"""Tests for MCP tools exposed through the synchronous ToolRegistry."""

from __future__ import annotations

import os
import subprocess
import sys
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
