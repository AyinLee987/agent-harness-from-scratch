"""Persistent MCP client sessions bridged into the synchronous agent runtime.

The bridge is one background thread owning one asyncio event loop, with a
queue in front of it: synchronous callers (the ReAct loop's tool dispatch,
on whatever thread it happens to be running) hand work in and block on a
``concurrent.futures.Future``. That shape is deliberate -- it keeps the
async-ness entirely inside this module, so nothing above it in the stack
(``ToolRegistry``, ``ToolDispatcher``, ``ReActLoop``) has to become async
to use an MCP tool.

Calls are dispatched as independent tasks on that loop rather than awaited
one at a time, so N Workers calling MCP tools in parallel actually overlap;
per-server concurrency is bounded by ``max_concurrent_calls`` and the wait
for a slot by ``slot_wait_timeout``, both on :class:`MCPServerConfig`.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, Callable, Dict, List, Optional, Set

from mcp import Client, StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from .config import MCPServerConfig
from .errors import MCPConnectionError, MCPManagerClosedError, MCPToolCallError
from .tool import MCPTool, format_mcp_result
from ..observability import get_logger, log_event


ClientFactory = Callable[[MCPServerConfig], Any]
logger = get_logger(__name__)


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
        # Created on the worker's own loop (an asyncio.Semaphore binds to
        # the loop that first awaits it), so this stays empty until then.
        self._semaphores: Dict[str, asyncio.Semaphore] = {}
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
        started = time.perf_counter()
        log_event(
            logger,
            logging.INFO,
            "mcp.connect.started",
            server_names=[config.name for config in self.configs],
            server_count=len(self.configs),
        )
        self._start_worker()
        timeout = sum(config.connect_timeout for config in self.configs) + 5.0
        try:
            discovered = self._submit("connect", None, timeout)
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "mcp.connect.failed",
                server_names=[config.name for config in self.configs],
                error_type=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                exc_info=True,
            )
            self.close()
            if isinstance(exc, MCPConnectionError):
                raise
            raise MCPConnectionError(f"Could not connect MCP servers: {exc}") from exc
        self._tools = discovered
        self._connected = True
        log_event(
            logger,
            logging.INFO,
            "mcp.connect.completed",
            server_names=[config.name for config in self.configs],
            tool_count=len(discovered),
            tool_names=[tool.name for tool in discovered],
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
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
        started = time.perf_counter()
        qualified_name = f"{server_name}__{tool_name}"
        log_event(
            logger,
            logging.INFO,
            "mcp.tool.started",
            server_name=server_name,
            tool_name=qualified_name,
            argument_keys=sorted(arguments.keys()),
        )
        # The worker enforces the real deadlines (slot wait, then call);
        # this caller-side timeout is only a backstop for a worker thread
        # that died without completing the future, so it must be the sum of
        # both plus slack. Setting it to `call_timeout` alone -- as the
        # first version did -- reported a *false* timeout for any call that
        # had merely been waiting its turn and had not yet been sent.
        budget = config.slot_wait_timeout + config.call_timeout + 5.0
        try:
            result = self._submit("call", payload, budget)
        except FutureTimeout as exc:
            log_event(
                logger,
                logging.WARNING,
                "mcp.tool.failed",
                server_name=server_name,
                tool_name=qualified_name,
                error_class="timeout",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            raise MCPToolCallError(
                f"MCP tool {qualified_name} timed out."
            ) from exc
        except MCPToolCallError:
            log_event(
                logger,
                logging.WARNING,
                "mcp.tool.failed",
                server_name=server_name,
                tool_name=qualified_name,
                error_class="classified",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                exc_info=True,
            )
            raise
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "mcp.tool.failed",
                server_name=server_name,
                tool_name=qualified_name,
                error_class="unexpected",
                error_type=type(exc).__name__,
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                exc_info=True,
            )
            raise MCPToolCallError(
                f"MCP tool {qualified_name} failed: {exc}"
            ) from exc
        formatted = format_mcp_result(result, config.max_output_chars)
        log_event(
            logger,
            logging.INFO,
            "mcp.tool.completed",
            server_name=server_name,
            tool_name=qualified_name,
            output_chars=len(formatted),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return formatted

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
        log_event(logger, logging.INFO, "mcp.closed")

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

    async def _run_call(
        self, sessions: Dict[str, Any], payload: Any, future: Future
    ) -> None:
        """Execute one tool call, bounded by its server's slot and call limits."""

        server_name, tool_name, arguments = payload
        config = self._configs_by_name[server_name]
        semaphore = self._semaphores[server_name]
        waited = 0.0
        queued_at = time.monotonic()
        try:
            await asyncio.wait_for(
                semaphore.acquire(), timeout=config.slot_wait_timeout
            )
        except asyncio.TimeoutError:
            future.set_exception(
                MCPToolCallError(
                    f"MCP tool {server_name}__{tool_name} waited "
                    f"{config.slot_wait_timeout}s for a free slot on server "
                    f"{server_name!r} (max_concurrent_calls="
                    f"{config.max_concurrent_calls}) and was never sent."
                )
            )
            return
        except asyncio.CancelledError:
            future.set_exception(
                MCPToolCallError(f"MCP tool {server_name}__{tool_name} was cancelled.")
            )
            raise

        waited = time.monotonic() - queued_at
        try:
            result = await asyncio.wait_for(
                sessions[server_name].call_tool(
                    tool_name,
                    arguments,
                    read_timeout_seconds=config.call_timeout,
                ),
                timeout=config.call_timeout,
            )
            future.set_result(result)
        except asyncio.TimeoutError:
            future.set_exception(
                MCPToolCallError(
                    f"MCP tool {server_name}__{tool_name} exceeded its "
                    f"{config.call_timeout}s call timeout "
                    f"(after {waited:.1f}s waiting for a slot)."
                )
            )
        except asyncio.CancelledError:
            future.set_exception(
                MCPToolCallError(f"MCP tool {server_name}__{tool_name} was cancelled.")
            )
            raise
        except Exception as exc:
            future.set_exception(exc)
        finally:
            semaphore.release()

    async def _worker(self) -> None:
        stack = AsyncExitStack()
        sessions: Dict[str, Any] = {}
        in_flight: Set[asyncio.Task] = set()
        self._semaphores = {
            config.name: asyncio.Semaphore(config.max_concurrent_calls)
            for config in self.configs
        }
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
                # Dispatched as its own task, not awaited here: awaiting it
                # inline made this loop a global serialization point, so two
                # Workers calling MCP tools at the same time took turns --
                # and the second one's caller-side timeout ran while it was
                # still queued, reporting a timeout for a call that had
                # never been sent.
                task = asyncio.create_task(
                    self._run_call(sessions, payload, future)
                )
                in_flight.add(task)
                task.add_done_callback(in_flight.discard)
            elif kind == "close":
                for task in list(in_flight):
                    task.cancel()
                if in_flight:
                    await asyncio.gather(*in_flight, return_exceptions=True)
                try:
                    await stack.aclose()
                    future.set_result(None)
                except Exception as exc:
                    future.set_exception(exc)
                break
