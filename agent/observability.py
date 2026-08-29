"""Structured, context-aware logging for the agent runtime.

The module intentionally uses only the Python standard library.  Application
code emits stable event names and metadata; prompts, tool arguments, and model
outputs are not logged by default.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, TextIO


_CONTEXT: ContextVar[Dict[str, Any]] = ContextVar("agent_log_context", default={})
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "password",
    "secret",
    "access_token",
    "refresh_token",
}


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the framework's namespace."""

    return logging.getLogger(name if name.startswith("agent") else f"agent.{name}")


@contextmanager
def bind_log_context(**fields: Any) -> Iterator[None]:
    """Temporarily add correlation fields to all logs in this context."""

    merged = dict(_CONTEXT.get())
    for key, value in fields.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    token = _CONTEXT.set(merged)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def current_log_context() -> Dict[str, Any]:
    """Return a copy of the active correlation context."""

    return dict(_CONTEXT.get())


def sanitize(value: Any, *, max_chars: Optional[int] = None) -> Any:
    """Make a value JSON-safe, redact secrets, and bound field size."""

    limit = max_chars or _env_int("AGENT_LOG_MAX_FIELD_CHARS", 1000)
    if isinstance(value, Mapping):
        clean: Dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name.lower() in _SENSITIVE_KEYS:
                clean[name] = "[REDACTED]"
            else:
                clean[name] = sanitize(item, max_chars=limit)
        return clean
    if isinstance(value, (list, tuple, set)):
        return [sanitize(item, max_chars=limit) for item in value]
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return text if len(text) <= limit else f"{text[:limit]}...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return sanitize(str(value), max_chars=limit)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    message: Optional[str] = None,
    *,
    exc_info: Any = None,
    **fields: Any,
) -> None:
    """Emit one structured event without leaking arbitrary object reprs."""

    logger.log(
        level,
        message or event,
        extra={"event": event, "event_fields": sanitize(fields)},
        exc_info=exc_info,
    )


class JsonFormatter(logging.Formatter):
    """One JSON object per line, suitable for local files and log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }
        payload.update(sanitize(_CONTEXT.get()))
        payload.update(sanitize(getattr(record, "event_fields", {})))
        if record.exc_info:
            payload["exception"] = _safe_exception(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    """Compact human-readable formatter retaining structured fields."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created).astimezone().isoformat(
            timespec="milliseconds"
        )
        metadata = dict(_CONTEXT.get())
        metadata.update(getattr(record, "event_fields", {}))
        suffix = f" {json.dumps(sanitize(metadata), ensure_ascii=False)}" if metadata else ""
        text = (
            f"{timestamp} {record.levelname:<8} {record.name} "
            f"[{getattr(record, 'event', 'log')}] {record.getMessage()}{suffix}"
        )
        if record.exc_info:
            text += "\n" + json.dumps(
                _safe_exception(record.exc_info), ensure_ascii=False
            )
        return text


def configure_logging(
    *,
    level: Optional[str] = None,
    format_name: Optional[str] = None,
    log_file: Optional[str] = None,
    stream: Optional[TextIO] = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``agent`` logger from arguments or ``AGENT_LOG_*`` env vars.

    Calling this repeatedly is safe.  ``AGENT_LOG_FILE`` enables a rotating
    file in addition to stderr; an empty value keeps logging console-only.
    """

    logger = logging.getLogger("agent")
    if getattr(logger, "_agent_configured", False) and not force:
        return logger
    if force:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    selected_level = (level or os.getenv("AGENT_LOG_LEVEL", "INFO")).upper()
    logger.setLevel(getattr(logging, selected_level, logging.INFO))
    logger.propagate = False
    formatter: logging.Formatter
    if (format_name or os.getenv("AGENT_LOG_FORMAT", "json")).lower() == "text":
        formatter = TextFormatter()
    else:
        formatter = JsonFormatter()

    console = logging.StreamHandler(stream or sys.stderr)
    console.setFormatter(formatter)
    logger.addHandler(console)

    selected_file = log_file if log_file is not None else os.getenv("AGENT_LOG_FILE", "")
    if selected_file:
        path = Path(selected_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        rotating = RotatingFileHandler(
            path,
            maxBytes=_env_int("AGENT_LOG_MAX_BYTES", 10 * 1024 * 1024),
            backupCount=_env_int("AGENT_LOG_BACKUP_COUNT", 5),
            encoding="utf-8",
        )
        rotating.setFormatter(formatter)
        logger.addHandler(rotating)

    logger._agent_configured = True  # type: ignore[attr-defined]
    return logger


class RequestLoggingMiddleware:
    """Pure ASGI middleware that keeps request context alive through SSE."""

    def __init__(self, app: Any) -> None:
        self.app = app
        self.logger = get_logger("server")

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }
        candidate = headers.get("x-request-id", "").strip()
        request_id = candidate[:128] if candidate else uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500

        async def send_with_request_id(message: Dict[str, Any]) -> None:
            nonlocal status_code
            if message.get("type") == "http.response.start":
                status_code = int(message.get("status", 500))
                response_headers = list(message.get("headers", []))
                response_headers.append((b"x-request-id", request_id.encode("ascii", "ignore")))
                message["headers"] = response_headers
            await send(message)

        with bind_log_context(request_id=request_id):
            log_event(
                self.logger,
                logging.INFO,
                "http.request.started",
                method=scope.get("method"),
                path=scope.get("path"),
            )
            try:
                await self.app(scope, receive, send_with_request_id)
            except Exception:
                log_event(
                    self.logger,
                    logging.ERROR,
                    "http.request.failed",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    exc_info=True,
                )
                raise
            finally:
                log_event(
                    self.logger,
                    logging.INFO,
                    "http.request.completed",
                    status_code=status_code,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                )


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _safe_exception(exc_info: Any) -> Dict[str, Any]:
    """Serialize stack locations while omitting potentially sensitive messages."""

    exc_type, _exc, tb = exc_info
    frames = traceback.extract_tb(tb) if tb is not None else []
    return {
        "type": getattr(exc_type, "__name__", str(exc_type)),
        "frames": [
            {"file": frame.filename, "line": frame.lineno, "function": frame.name}
            for frame in frames
        ],
    }
