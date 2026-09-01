import asyncio
import io
import json
import logging

from agent import (
    JsonFormatter,
    RequestLoggingMiddleware,
    bind_log_context,
    configure_logging,
    current_log_context,
    get_logger,
    log_event,
    run_log_file,
)


def test_json_logging_adds_context_and_redacts_secrets():
    output = io.StringIO()
    configure_logging(
        level="INFO", format_name="json", stream=output, log_file="", force=True
    )

    with bind_log_context(request_id="req-1", root_run_id="root-1"):
        log_event(
            get_logger("test"),
            logging.INFO,
            "test.event",
            api_key="never-log-me",
            headers={"authorization": "Bearer secret", "accept": "json"},
            tokens=42,
        )

    payload = json.loads(output.getvalue())
    assert payload["event"] == "test.event"
    assert payload["request_id"] == "req-1"
    assert payload["root_run_id"] == "root-1"
    assert payload["api_key"] == "[REDACTED]"
    assert payload["headers"]["authorization"] == "[REDACTED]"
    assert payload["tokens"] == 42
    assert "never-log-me" not in output.getvalue()
    assert "Bearer secret" not in output.getvalue()


def test_log_context_is_nested_and_restored():
    assert current_log_context() == {}
    with bind_log_context(request_id="outer"):
        assert current_log_context() == {"request_id": "outer"}
        with bind_log_context(run_id="run-1"):
            assert current_log_context() == {
                "request_id": "outer",
                "run_id": "run-1",
            }
        assert current_log_context() == {"request_id": "outer"}
    assert current_log_context() == {}


def test_request_middleware_correlates_stream_and_returns_header():
    output = io.StringIO()
    configure_logging(
        level="INFO", format_name="json", stream=output, log_file="", force=True
    )
    seen_context = {}

    async def inner(scope, receive, send):
        seen_context.update(current_log_context())
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [(b"x-request-id", b"client-request")],
    }
    asyncio.run(RequestLoggingMiddleware(inner)(scope, receive, send))

    start = next(item for item in messages if item["type"] == "http.response.start")
    assert (b"x-request-id", b"client-request") in start["headers"]
    assert seen_context["request_id"] == "client-request"
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [event["event"] for event in events] == [
        "http.request.started",
        "http.request.completed",
    ]
    assert events[-1]["status_code"] == 204


def test_json_formatter_truncates_large_fields(monkeypatch):
    monkeypatch.setenv("AGENT_LOG_MAX_FIELD_CHARS", "8")
    record = logging.LogRecord("agent.test", logging.INFO, "", 0, "ok", (), None)
    record.event = "truncate"
    record.event_fields = {"value": "abcdefghijklmnop"}
    payload = json.loads(JsonFormatter().format(record))
    assert payload["value"] == "abcdefgh...[truncated]"


def test_run_log_file_writes_only_matching_run(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "true")
    logger = configure_logging(
        level="INFO", format_name="json", stream=io.StringIO(), log_file="", force=True
    )

    with bind_log_context(run_id="run-a"), run_log_file(
        "run-a", log_dir=str(tmp_path)
    ) as path_a:
        log_event(logger, logging.INFO, "a.only", note="from run a")
        with bind_log_context(run_id="run-b"):
            log_event(logger, logging.INFO, "b.only", note="from run b")

    files = sorted(tmp_path.glob("*_run-a.log"))
    assert len(files) == 1
    assert path_a == files[0]
    lines = files[0].read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert [e["event"] for e in events] == ["a.only"]
    assert events[0]["run_id"] == "run-a"


def test_run_log_file_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_LOG_PER_RUN", "false")
    logger = configure_logging(
        level="INFO", format_name="json", stream=io.StringIO(), log_file="", force=True
    )

    with bind_log_context(run_id="run-c"), run_log_file(
        "run-c", log_dir=str(tmp_path)
    ) as path_c:
        assert path_c is None
        log_event(logger, logging.INFO, "c.only")

    assert list(tmp_path.glob("*.log")) == []


def test_exception_logs_keep_stack_but_omit_exception_message():
    output = io.StringIO()
    configure_logging(
        level="INFO", format_name="json", stream=output, log_file="", force=True
    )
    try:
        raise RuntimeError("password-like-value-must-not-leak")
    except RuntimeError:
        log_event(
            get_logger("test"),
            logging.ERROR,
            "test.failed",
            exc_info=True,
        )

    payload = json.loads(output.getvalue())
    assert payload["exception"]["type"] == "RuntimeError"
    assert payload["exception"]["frames"]
    assert "password-like-value-must-not-leak" not in output.getvalue()
