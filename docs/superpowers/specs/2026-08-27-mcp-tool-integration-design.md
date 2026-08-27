# Generic MCP Tool Integration Design

## Goal

Allow `ReActAgent` to discover tools from one or more MCP servers and use them
through the existing `ToolRegistry` exactly like local `BaseTool`
implementations. Fetch MCP is the first example; the adapter is server-agnostic.

## Scope

The first release supports MCP tools over stdio and Streamable HTTP:
configuration, persistent connection lifecycle, discovery, schema adaptation,
tool calls, result conversion, namespacing, errors, timeouts, and cleanup.

Resources, prompts, sampling, elicitation, roots, OAuth UI, dynamic tool-list
notifications, automatic reconnect, and legacy SSE are out of scope. The
official MCP Python SDK v2 is used, raising the minimum Python version to 3.10.

## Architecture

MCP is a tool-integration layer between `ToolRegistry` and external servers:

1. Application bootstrap creates `MCPServerConfig` objects.
2. `MCPManager` opens persistent sessions and calls `list_tools`.
3. Each discovered tool becomes an `MCPTool(BaseTool)`.
4. These proxies are registered in the existing `ToolRegistry`.
5. The LLM sees ordinary OpenAI-style function schemas.
6. `ToolDispatcher` dispatches without knowing a tool is remote.
7. `MCPTool` delegates to `MCPManager.call()`, invoking MCP `tools/call`.
8. MCP content becomes an observation and passes through `ToolOutputGuard`.

The synchronous `ReActAgent.run()` API remains unchanged. A dedicated
background event loop owns async MCP clients and sessions. Synchronous
`MCPTool.run()` submits work to that loop with a bounded timeout.

## Components

### MCPServerConfig

Each server has a unique `name`, transport `stdio` or `streamable_http`,
and positive connection/call timeouts. Stdio has `command`, `args`, optional
`env` and `cwd`; HTTP has `url` and optional headers.

Validation rejects missing transport fields, duplicate names, unsupported
transports, empty commands or URLs, and invalid timeouts. Secrets never appear
in `repr` or integration errors.

### MCPManager

```python
with MCPManager(configs) as manager:
    tools = manager.tools()
    result = manager.call(server_name, remote_tool_name, arguments)
```

Entering starts the loop, connects all servers, and completes discovery. All
async objects are created, used, and closed on the same loop. Startup is atomic:
if any server fails, opened sessions close and no partial list is returned.
`close()` is idempotent, closes sessions and stdio subprocesses, stops the
thread, and rejects later calls.

### MCPTool

`MCPTool` stores the namespaced local name, remote server/tool names,
description, unchanged MCP `inputSchema`, and its manager.
`parameters_schema()` returns that schema; `run(**kwargs)` invokes the manager
and returns normalized text.

Local names are `<server>__<tool>`, normalized for model function names. Any
collision after normalization fails startup instead of overwriting a tool.

### ToolRegistry

No MCP branch is added to `ToolRegistry` or `ToolDispatcher`.
`register_many()` is added, and `register()` rejects duplicates rather than
silently replacing a trusted local or remote tool.

### Result conversion

- Text blocks are concatenated in order.
- Structured content becomes UTF-8 JSON unless equivalent text exists.
- Image, audio, embedded-resource, and resource-link blocks become concise
  metadata placeholders because observations are currently text-only.
- MCP error results raise `MCPToolCallError`; the dispatcher produces an
  `ERROR:` observation.
- Empty success results become `(no content)`.
- An output-size ceiling is applied before model ingestion.

## Failure Behaviour

- Invalid configuration fails before starting a process or network request.
- Errors identify the server without leaking env values or headers.
- Calls on disconnected or closed managers fail immediately.
- Connect and call operations have explicit timeouts.
- Shutdown attempts every session even if one close operation fails.
- Calls are not retried automatically because MCP tools may have side effects.

## Public API and Fetch Example

The package exports `MCPManager`, `MCPServerConfig`, and `MCPTool`.

```python
config = MCPServerConfig.stdio(
    name="fetch",
    command="uvx",
    args=["mcp-server-fetch"],
)

with MCPManager([config]) as mcp:
    registry = ToolRegistry([calculator])
    registry.register_many(mcp.tools())
    agent = ReActAgent(llm=llm, tools=registry)
    result = agent.run("Fetch and summarize https://example.com")
```

The exact Fetch command will be verified during implementation and documented
with an equivalent Python-module launch when available.

## Files

- `agent/mcp/config.py`: validated server configuration.
- `agent/mcp/client.py`: manager, loop bridge, transports, lifecycle.
- `agent/mcp/tool.py`: proxy tool and result conversion.
- `agent/mcp/errors.py`: integration exceptions.
- `agent/mcp/__init__.py`, `agent/__init__.py`: public exports.
- `agent/tools.py`: duplicate protection and `register_many()`.
- `requirements.txt`: MCP dependency and Python requirement.
- `examples/mcp_fetch.py`: runnable Fetch example.
- `tests/test_mcp.py`: deterministic integration tests.
- `README.md`: setup, architecture, usage, and security notes.

## Testing

Tests use an in-process fake MCP server or SDK in-memory transport, not the
internet or Fetch executable. They cover configuration, discovery, schemas,
arguments, result types, errors, collisions, partial-startup cleanup, timeouts,
and shutdown. Existing tests must remain green.

An optional manual smoke test starts Fetch MCP, discovers `fetch__fetch`, and
retrieves a public page.

## Compatibility and Security

The synchronous agent and local tools remain compatible. Python 3.10+ is
required. MCP servers are trusted configuration, but returned content remains
untrusted and continues through `ToolOutputGuard`. Only explicit environment
values and headers are passed; diagnostics redact them. Tool collisions fail
closed, and side-effecting calls are never retried implicitly.
