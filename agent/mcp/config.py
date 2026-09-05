"""Validated configuration for MCP server connections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class MCPServerConfig:
    """Connection settings for one MCP server."""

    name: str
    transport: Literal["stdio", "streamable_http"]
    command: Optional[str] = None
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict, repr=False)
    cwd: Optional[str] = None
    url: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict, repr=False)
    connect_timeout: float = 30.0
    call_timeout: float = 60.0
    max_output_chars: int = 100_000
    #: How many calls this server may be running at once. Concurrency is
    #: bounded per server rather than globally because the limit belongs to
    #: whatever is on the other end of the transport -- a local stdio
    #: process and a hosted HTTP endpoint tolerate very different loads.
    max_concurrent_calls: int = 4
    #: Ceiling on how long a call may wait for one of those slots. Separate
    #: from ``call_timeout`` so a wait can be reported as a wait ("the
    #: server was busy") instead of masquerading as the tool being slow.
    slot_wait_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("MCP server name must be non-empty.")
        if self.transport not in ("stdio", "streamable_http"):
            raise ValueError(f"Unsupported MCP transport: {self.transport!r}.")
        if self.connect_timeout <= 0 or self.call_timeout <= 0:
            raise ValueError("MCP timeouts must be positive.")
        if self.slot_wait_timeout <= 0:
            raise ValueError("MCP timeouts must be positive.")
        if self.max_concurrent_calls < 1:
            raise ValueError("max_concurrent_calls must be at least 1.")
        if self.max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive.")
        if self.transport == "stdio":
            if not self.command or not self.command.strip():
                raise ValueError("stdio MCP servers require a non-empty command.")
            if self.url is not None:
                raise ValueError("stdio MCP servers cannot define a URL.")
        else:
            if not self.url:
                raise ValueError("Streamable HTTP MCP servers require a URL.")
            parsed = urlparse(self.url)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("MCP HTTP URL must use http:// or https://.")
            if self.command is not None:
                raise ValueError("HTTP MCP servers cannot define a command.")

    @classmethod
    def stdio(
        cls,
        *,
        name: str,
        command: str,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        connect_timeout: float = 30.0,
        call_timeout: float = 60.0,
        max_output_chars: int = 100_000,
        max_concurrent_calls: int = 4,
        slot_wait_timeout: float = 30.0,
    ) -> "MCPServerConfig":
        return cls(
            name=name,
            transport="stdio",
            command=command,
            args=list(args or []),
            env=dict(env or {}),
            cwd=cwd,
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            max_output_chars=max_output_chars,
            max_concurrent_calls=max_concurrent_calls,
            slot_wait_timeout=slot_wait_timeout,
        )

    @classmethod
    def http(
        cls,
        *,
        name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        connect_timeout: float = 30.0,
        call_timeout: float = 60.0,
        max_output_chars: int = 100_000,
        max_concurrent_calls: int = 4,
        slot_wait_timeout: float = 30.0,
    ) -> "MCPServerConfig":
        return cls(
            name=name,
            transport="streamable_http",
            url=url,
            headers=dict(headers or {}),
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
            max_output_chars=max_output_chars,
            max_concurrent_calls=max_concurrent_calls,
            slot_wait_timeout=slot_wait_timeout,
        )
