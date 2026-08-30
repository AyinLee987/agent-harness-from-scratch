from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

from app import server
from agent import (
    ListFilesTool, LocalToolConfig, ReadFileTool, RecoverableToolError,
    RunCommandTool, WriteFileTool, create_local_tools,
)


def _config(root: Path, **overrides):
    values = {
        "workspace_root": root,
        "allowed_commands": (Path(sys.executable).name,),
        "command_timeout_seconds": 2,
        "max_command_output_bytes": 128,
    }
    values.update(overrides)
    return LocalToolConfig(**values)


def test_read_file_supports_ranges_and_blocks_traversal(tmp_path: Path):
    (tmp_path / "note.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    tool = ReadFileTool(_config(tmp_path))
    output = tool.run(path="note.txt", start_line=2, end_line=2)
    assert output.endswith("two\n")
    with pytest.raises(RecoverableToolError, match="outside workspace"):
        tool.run(path=str(tmp_path.parent / "outside.txt"))


def test_write_file_is_atomic_and_requires_explicit_overwrite(tmp_path: Path):
    tool = WriteFileTool(_config(tmp_path))
    created = tool.run(path="nested/note.txt", content="first", create_parent_dirs=True)
    assert "sha256=" in created
    target = tmp_path / "nested" / "note.txt"
    assert target.read_text(encoding="utf-8") == "first"
    with pytest.raises(RecoverableToolError, match="overwrite=true"):
        tool.run(path="nested/note.txt", content="second")
    digest = hashlib.sha256(b"first").hexdigest()
    tool.run(path="nested/note.txt", content="second", overwrite=True, expected_sha256=digest)
    assert target.read_text(encoding="utf-8") == "second"
    assert not list(target.parent.glob(".agent-write-*"))


def test_write_file_detects_concurrent_change(tmp_path: Path):
    target = tmp_path / "note.txt"
    target.write_text("new value", encoding="utf-8")
    tool = WriteFileTool(_config(tmp_path))
    with pytest.raises(RecoverableToolError, match="changed since"):
        tool.run(
            path="note.txt", content="replacement", overwrite=True,
            expected_sha256=hashlib.sha256(b"old value").hexdigest(),
        )


def test_list_files_glob_is_workspace_scoped(tmp_path: Path):
    (tmp_path / "a.py").write_text("", encoding="utf-8")
    (tmp_path / "b.txt").write_text("", encoding="utf-8")
    tool = ListFilesTool(_config(tmp_path))
    assert tool.run(pattern="*.py") == "a.py"
    with pytest.raises(RecoverableToolError, match="cannot contain"):
        tool.run(pattern="../*")


def test_run_command_uses_argv_allowlist_and_redacts_process_environment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    executable = Path(sys.executable).name
    tool = RunCommandTool(_config(tmp_path))
    output = tool.run(
        argv=[executable, "-c", "import os; print(os.getenv('OPENAI_API_KEY'))"]
    )
    assert "EXIT 0" in output
    assert "None" in output
    assert "must-not-leak" not in output
    with pytest.raises(RecoverableToolError, match="not allowed"):
        tool.run(argv=["not-allowed", "--version"])


def test_run_command_times_out_and_truncates_output(tmp_path: Path):
    executable = Path(sys.executable).name
    timeout_tool = RunCommandTool(_config(tmp_path, command_timeout_seconds=0.1))
    with pytest.raises(RecoverableToolError, match="timed out"):
        timeout_tool.run(argv=[executable, "-c", "import time; time.sleep(2)"])
    output_tool = RunCommandTool(_config(tmp_path, max_command_output_bytes=16))
    output = output_tool.run(argv=[executable, "-c", "print('x' * 100)"])
    assert "[output truncated]" in output


def test_factory_and_server_registration_are_explicit(monkeypatch, tmp_path: Path):
    tools = create_local_tools(_config(tmp_path), include_files=True, include_cli=True)
    assert {tool.name for tool in tools} == {"read_file", "write_file", "list_files", "run_command"}
    monkeypatch.setenv("ENABLE_LOCAL_FILE_TOOLS", "1")
    monkeypatch.setenv("ENABLE_LOCAL_CLI", "0")
    monkeypatch.setenv("AGENT_WORKSPACE_ROOT", str(tmp_path))
    registry = server.build_registry()
    assert {"read_file", "write_file", "list_files"} <= set(registry.names())
    assert "run_command" not in registry.names()
