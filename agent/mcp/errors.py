"""Errors raised by the MCP tool integration layer."""

from ..errors import RecoverableToolError, ToolCallError


class MCPIntegrationError(ToolCallError):
    """Base error for MCP configuration, lifecycle, and calls."""


class MCPConnectionError(MCPIntegrationError):
    """An MCP server could not be connected or discovered."""


class MCPToolCallError(MCPIntegrationError, RecoverableToolError):
    """An MCP tool returned an error or could not be called."""


class MCPManagerClosedError(MCPIntegrationError):
    """An operation was attempted after the manager closed."""
