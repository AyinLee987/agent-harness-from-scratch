"""Persistent MCP client sessions bridged into the synchronous agent runtime."""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Callable, Dict, List, Optional

from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from .config import MCPServerConfig
from .errors import MCPConnectionError, MCPManagerClosedError, MCPToolCallError
from .tool import MCPTool, format_mcp_result


ClientFactory = Callable[[MCPServerConfig], Any]


def _exception_details(exc: BaseException) -> str:
    """Flatten ExceptionGroup-like errors into a useful one-line message."""

    nested = getattr(exc, "exceptions", None)
    if nested:
        children = "; ".join(_exception_details(child) for child in nested)
        return f"{type(exc).__name__}: {exc} [{children}]"
    return f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def _http_client_context(config: MCPServerConfig):
    if config.headers:
        import httpx2

        async with httpx2.AsyncClient(headers=config.headers) as http_client:
            transport = streamable_http_client(config.url, http_client=http_client)
            async with Client(
                transport, read_timeout_seconds=config.call_timeout
            ) as client:
                yield client
    else:
        async with Client(
            config.url, read_timeout_seconds=config.call_timeout
        ) as client:
            yield client


def _default_client_factory(config: MCPServerConfig):
    if config.transport == "stdio":
        parameters = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env or None,
            cwd=config.cwd,
            encoding="utf-8",
            encoding_error_handler="replace",
        )
        # Stdio servers commonly implement the classic MCP handshake. Avoid the
        # v2 auto-discovery probe, which older servers log as a validation error
        # before the SDK falls back successfully.
        return Client(
            parameters,
            read_timeout_seconds=config.call_timeout,
            mode="legacy",
        )
    return _http_client_context(config)


class MCPManager:
    """Own persistent MCP sessions and expose their tools synchronously."""

    def __init__(
        self,
        configs: List[MCPServerConfig],
        *,
        client_factory: Optional[ClientFactory] = None,
    ) -> None:
        names = [config.name for config in configs]
        if len(names) != len(set(names)):
            raise ValueError("MCP server names must be unique.")
        self.configs = list(configs)
        self._configs_by_name = {config.name: config for config in configs}
        self._client_factory = client_factory or _default_client_factory
        self._tools: List[MCPTool] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._queue: Optional[asyncio.Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._connected = False
        self._closed = False

    def __enter__(self) -> "MCPManager":
        self.connect_all()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def connect_all(self) -> List[MCPTool]:
        if self._closed:
            raise MCPManagerClosedError("MCP manager is closed.")
        if self._connected:
            return self.tools()
        self._start_worker()
        timeout = sum(config.connect_timeout for config in self.configs) + 5.0
        try:
            discovered = self._submit("connect", None, timeout)
        except Exception as exc:
            self.close()
            if isinstance(exc, MCPConnectionError):
                raise
            raise MCPConnectionError(f"Could not connect MCP servers: {exc}") from exc
        self._tools = discovered
        self._connected = True
        return self.tools()

    def tools(self) -> List[MCPTool]:
        if not self._connected:
            raise MCPConnectionError("MCP servers are not connected.")
        return list(self._tools)

    def call(
        self, server_name: str, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        if self._closed:
            raise MCPManagerClosedError("MCP manager is closed.")
        if not self._connected:
            raise MCPConnectionError("MCP servers are not connected.")
        config = self._configs_by_name.get(server_name)
        if config is None:
            raise MCPToolCallError(f"Unknown MCP server: {server_name!r}.")
        payload = (server_name, tool_name, dict(arguments))
        try:
            result = self._submit("call", payload, config.call_timeout + 1.0)
        except FutureTimeout as exc:
            raise MCPToolCallError(
                f"MCP tool {server_name}__{tool_name} timed out."
            ) from exc
        except MCPToolCallError:
            raise
        except Exception as exc:
            raise MCPToolCallError(
                f"MCP tool {server_name}__{tool_name} failed: {exc}"
            ) from exc
        return format_mcp_result(result, config.max_output_chars)

    def close(self) -> None:
        if self._closed:
            return
        if self._thread is not None and self._thread.is_alive():
            try:
                timeout = sum(config.connect_timeout for config in self.configs) + 5.0
                self._submit("close", None, timeout)
            finally:
                self._thread.join(timeout=5.0)
        self._connected = False
        self._closed = True

    def _start_worker(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._thread_main, name="mcp-manager", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise MCPConnectionError("MCP background worker did not start.")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._queue = asyncio.Queue()
        self._ready.set()
        try:
            loop.run_until_complete(self._worker())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    def _submit(self, kind: str, payload: Any, timeout: float):
        if self._loop is None or self._queue is None:
            raise MCPConnectionError("MCP background worker is unavailable.")
        future: Future = Future()
        self._loop.call_soon_threadsafe(
            self._queue.put_nowait, (kind, payload, future)
        )
        return future.result(timeout=timeout)

    async def _worker(self) -> None:
        stack = AsyncExitStack()
        sessions: Dict[str, Any] = {}
        while True:
            kind, payload, future = await self._queue.get()
            if kind == "connect":
                candidate_stack = AsyncExitStack()
                candidate_sessions: Dict[str, Any] = {}
                discovered: List[MCPTool] = []
                seen_names = set()
                try:
                    for config in self.configs:
                        try:
                            client = await asyncio.wait_for(
                                candidate_stack.enter_async_context(
                                    self._client_factory(config)
                                ),
                                timeout=config.connect_timeout,
                            )
                            listing = await asyncio.wait_for(
                                client.list_tools(), timeout=config.connect_timeout
                            )
                        except Exception as exc:
                            raise MCPConnectionError(
                                f"MCP server {config.name!r} discovery failed: "
                                f"{_exception_details(exc)}"
                            ) from exc
                        candidate_sessions[config.name] = client
                        for remote in listing.tools:
                            proxy = MCPTool(
                                manager=self,
                                server_name=config.name,
                                remote_name=remote.name,
                                description=remote.description or remote.name,
                                input_schema=getattr(
                                    remote,
                                    "input_schema",
                                    getattr(remote, "inputSchema", {}),
                                ),
                            )
                            if proxy.name in seen_names:
                                raise ValueError(
                                    f"Duplicate normalized MCP tool: {proxy.name!r}."
                                )
                            seen_names.add(proxy.name)
                            discovered.append(proxy)
                    stack = candidate_stack.pop_all()
                    sessions = candidate_sessions
                    future.set_result(discovered)
                except Exception as exc:
                    cleanup_error = None
                    try:
                        await candidate_stack.aclose()
                    except Exception as close_exc:
                        cleanup_error = close_exc
                    if isinstance(exc, MCPConnectionError):
                        error = exc
                    else:
                        error = MCPConnectionError(
                            f"MCP discovery failed: {_exception_details(exc)}"
                        )
                    if cleanup_error is not None:
                        error = MCPConnectionError(
                            f"{error}; cleanup failed: "
                            f"{_exception_details(cleanup_error)}"
                        )
                    future.set_exception(error)
            elif kind == "call":
                server_name, tool_name, arguments = payload
                config = self._configs_by_name[server_name]
                try:
                    session = sessions[server_name]
                    result = await asyncio.wait_for(
                        session.call_tool(
                            tool_name,
                            arguments,
                            read_timeout_seconds=config.call_timeout,
                        ),
                        timeout=config.call_timeout,
                    )
                    future.set_result(result)
                except Exception as exc:
                    future.set_exception(exc)
            elif kind == "close":
                try:
                    await stack.aclose()
                    future.set_result(None)
                except Exception as exc:
                    future.set_exception(exc)
                break
