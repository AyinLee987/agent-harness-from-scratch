"""Adapter from remote MCP tools to the harness' BaseTool interface."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, TYPE_CHECKING

from ..tools import BaseTool
from .errors import MCPToolCallError

if TYPE_CHECKING:
    from .client import MCPManager


def _attr(value: Any, snake_name: str, protocol_name: str, default: Any = None) -> Any:
    """Read either SDK v2 snake_case or protocol/v1 camelCase attributes."""

    return getattr(value, snake_name, getattr(value, protocol_name, default))


def normalize_tool_name(server_name: str, tool_name: str) -> str:
    """Return a model-compatible, namespaced function name."""

    raw = f"{server_name}__{tool_name}"
    normalized = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    if not normalized.strip("_-"):
        raise ValueError(f"MCP tool name {raw!r} cannot be normalized.")
    return normalized


def format_mcp_result(result: Any, max_chars: int = 100_000) -> str:
    """Convert MCP content blocks into the agent's text observation format."""

    parts: List[str] = []
    for block in getattr(result, "content", []) or []:
        kind = getattr(block, "type", "")
        if kind == "text":
            parts.append(str(block.text))
        elif kind == "image":
            parts.append(f"[image: {_attr(block, 'mime_type', 'mimeType', 'unknown')}]")
        elif kind == "audio":
            parts.append(f"[audio: {_attr(block, 'mime_type', 'mimeType', 'unknown')}]")
        elif kind == "resource_link":
            parts.append(f"[resource link: {getattr(block, 'uri', 'unknown')}]")
        elif kind == "resource":
            resource = getattr(block, "resource", None)
            uri = getattr(resource, "uri", "unknown")
            mime = _attr(resource, "mime_type", "mimeType")
            label = f"{uri} ({mime})" if mime else str(uri)
            parts.append(f"[embedded resource: {label}]")
        else:
            parts.append(f"[MCP content: {kind or 'unknown'}]")

    structured = _attr(result, "structured_content", "structuredContent")
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False, default=str))

    rendered = "\n".join(part for part in parts if part).strip() or "(no content)"
    if _attr(result, "is_error", "isError", False):
        raise MCPToolCallError(rendered)
    if len(rendered) > max_chars:
        omitted = len(rendered) - max_chars
        rendered = f"{rendered[:max_chars]}\n[truncated {omitted} characters]"
    return rendered


class MCPTool(BaseTool):
    """A local BaseTool proxy for one tool exposed by an MCP server."""

    def __init__(
        self,
        *,
        manager: "MCPManager",
        server_name: str,
        remote_name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> None:
        self.manager = manager
        self.server_name = server_name
        self.remote_name = remote_name
        self.name = normalize_tool_name(server_name, remote_name)
        self.description = description or f"MCP tool {remote_name} from {server_name}."
        self._input_schema = dict(input_schema)

    def parameters_schema(self) -> Dict[str, Any]:
        return dict(self._input_schema)

    def run(self, **kwargs: Any) -> str:
        return self.manager.call(self.server_name, self.remote_name, kwargs)
