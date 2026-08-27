"""Use the official Fetch MCP server as an ordinary ReAct agent tool.

Prerequisites:
    pip install -r requirements.txt
    pip install uv

Run:
    python examples/mcp_fetch.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import MockLLM, ReActAgent, ToolOutputGuard, ToolRegistry
from agent.mcp import MCPManager, MCPServerConfig, uv_tool_command


def main() -> None:
    command, args = uv_tool_command("mcp-server-fetch")
    fetch = MCPServerConfig.stdio(
        name="fetch",
        command=command,
        args=args,
        # Recommended by the Fetch project for reliable stdio on Windows.
        env={"PYTHONIOENCODING": "utf-8"},
        connect_timeout=90,
        call_timeout=60,
    )

    with MCPManager([fetch]) as mcp:
        registry = ToolRegistry()
        registry.register_many(mcp.tools())
        print(f"Discovered tools: {', '.join(registry.names())}")

        agent = ReActAgent(
            llm=MockLLM(),
            tools=registry,
            output_guard=ToolOutputGuard(),
            max_steps=4,
        )
        result = agent.run("Fetch https://example.com and summarize it.")
        called = [
            step["action"]["name"]
            for step in result.trajectory
            if step["action"] is not None
        ]
        print(f"Called tools: {', '.join(called)}")
        print(result.answer)


if __name__ == "__main__":
    main()
