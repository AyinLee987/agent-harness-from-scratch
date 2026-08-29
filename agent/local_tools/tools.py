"""Workspace-scoped file tools and an explicitly enabled local CLI tool."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

from ..errors import RecoverableToolError
from ..tools import BaseTool


@dataclass
class LocalToolConfig:
    workspace_root: Path | str
    max_read_bytes: int = 2 * 1024 * 1024
    max_write_bytes: int = 2 * 1024 * 1024
    max_command_output_bytes: int = 256 * 1024
    command_timeout_seconds: float = 30.0
    allowed_commands: Sequence[str] = field(
        default_factory=lambda: ("git", "rg", "python", "python.exe", "pytest")
    )
    allowed_environment: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve(strict=True)
        if not self.workspace_root.is_dir():
            raise ValueError("workspace_root must be an existing directory.")
        if min(self.max_read_bytes, self.max_write_bytes, self.max_command_output_bytes) <= 0:
            raise ValueError("Local tool byte limits must be positive.")
        if self.command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be positive.")


class _WorkspaceTool(BaseTool):
    def __init__(self, config: LocalToolConfig) -> None:
        self.config = config
        self.root = Path(config.workspace_root)

    def _resolve(self, value: str, *, must_exist: bool = False) -> Path:
        if not value or "\x00" in value:
            raise RecoverableToolError("Path must be a non-empty workspace path.")
        candidate = Path(value)
        candidate = candidate if candidate.is_absolute() else self.root / candidate
        try:
            resolved = candidate.resolve(strict=must_exist)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise RecoverableToolError(
                f"Path is unavailable or outside workspace {self.root}: {value}"
            ) from exc
        return resolved

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."


class ReadFileTool(_WorkspaceTool):
    name = "read_file"
    description = "Read a UTF-8 text file inside the configured workspace, optionally by line range."

    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "start_line": {"type": "integer", "description": "First 1-based line; default 1."},
                "end_line": {"type": "integer", "description": "Inclusive last line; 0 means EOF."},
            },
            "required": ["path"],
        }

    def run(self, path: str, start_line: int = 1, end_line: int = 0) -> str:
        if start_line < 1 or end_line < 0 or (end_line and end_line < start_line):
            raise RecoverableToolError("Require start_line >= 1 and end_line=0 or end_line >= start_line.")
        target = self._resolve(path, must_exist=True)
        if not target.is_file():
            raise RecoverableToolError(f"Not a regular file: {path}")
        selected: List[str] = []
        size = 0
        try:
            with target.open("r", encoding="utf-8", errors="strict") as handle:
                for line_number, line in enumerate(handle, 1):
                    if line_number < start_line:
                        continue
                    if end_line and line_number > end_line:
                        break
                    size += len(line.encode("utf-8"))
                    if size > self.config.max_read_bytes:
                        raise RecoverableToolError(
                            f"Selected content exceeds {self.config.max_read_bytes} bytes; request a smaller line range."
                        )
                    selected.append(line)
        except UnicodeDecodeError as exc:
            raise RecoverableToolError("read_file only supports UTF-8 text files.") from exc
        except OSError as exc:
            raise RecoverableToolError(f"Could not read {path}: {exc}") from exc
        return f"FILE {self._relative(target)} lines {start_line}-{end_line or 'EOF'}\n{''.join(selected)}"


class WriteFileTool(_WorkspaceTool):
    name = "write_file"
    description = "Atomically create or replace a UTF-8 text file inside the configured workspace."

    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace-relative file path."},
                "content": {"type": "string", "description": "Complete UTF-8 file content."},
                "overwrite": {"type": "boolean", "description": "Must be true to replace an existing file."},
                "create_parent_dirs": {"type": "boolean", "description": "Create missing workspace subdirectories."},
                "expected_sha256": {
                    "type": "string",
                    "description": "Optional current SHA-256; prevents overwriting a file changed since it was read.",
                },
            },
            "required": ["path", "content"],
        }

    def run(
        self,
        path: str,
        content: str,
        overwrite: bool = False,
        create_parent_dirs: bool = False,
        expected_sha256: str = "",
    ) -> str:
        data = content.encode("utf-8")
        if len(data) > self.config.max_write_bytes:
            raise RecoverableToolError(f"Content exceeds {self.config.max_write_bytes} bytes.")
        target = self._resolve(path)
        if target.exists() and not target.is_file():
            raise RecoverableToolError(f"Target is not a regular file: {path}")
        if target.exists() and not overwrite:
            raise RecoverableToolError("File exists; retry with overwrite=true after reading it.")
        if target.exists() and expected_sha256:
            current = hashlib.sha256(target.read_bytes()).hexdigest()
            if current.lower() != expected_sha256.lower():
                raise RecoverableToolError("File changed since it was read; read it again before overwriting.")
        parent = self._resolve(str(target.parent))
        if not parent.exists():
            if not create_parent_dirs:
                raise RecoverableToolError("Parent directory does not exist; set create_parent_dirs=true.")
            try:
                parent.mkdir(parents=True)
            except OSError as exc:
                raise RecoverableToolError(f"Could not create parent directories: {exc}") from exc
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix=".agent-write-", dir=parent)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, target)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        except OSError as exc:
            raise RecoverableToolError(f"Could not write {path}: {exc}") from exc
        digest = hashlib.sha256(data).hexdigest()
        return f"WROTE {self._relative(target)} bytes={len(data)} sha256={digest}"


class ListFilesTool(_WorkspaceTool):
    name = "list_files"
    description = "List files and directories under a workspace path using a glob pattern."

    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Workspace directory; default '.'."},
                "pattern": {"type": "string", "description": "Relative glob such as '*.py' or '**/*.md'."},
                "max_results": {"type": "integer", "description": "Maximum results, 1-1000."},
            },
            "required": [],
        }

    def run(self, path: str = ".", pattern: str = "*", max_results: int = 200) -> str:
        if not 1 <= max_results <= 1000:
            raise RecoverableToolError("max_results must be between 1 and 1000.")
        pattern_path = Path(pattern)
        if pattern_path.is_absolute() or ".." in pattern_path.parts:
            raise RecoverableToolError("Glob pattern must stay relative and cannot contain '..'.")
        directory = self._resolve(path, must_exist=True)
        if not directory.is_dir():
            raise RecoverableToolError(f"Not a directory: {path}")
        results: List[str] = []
        try:
            for item in directory.glob(pattern):
                try:
                    resolved = item.resolve(strict=True)
                    resolved.relative_to(self.root)
                except (OSError, ValueError):
                    continue
                results.append(self._relative(resolved) + ("/" if resolved.is_dir() else ""))
                if len(results) >= max_results:
                    break
        except (OSError, ValueError) as exc:
            raise RecoverableToolError(f"Could not list files: {exc}") from exc
        return "\n".join(results) if results else "(no matches)"


class RunCommandTool(_WorkspaceTool):
    name = "run_command"
    description = (
        "Run one allowlisted local executable in the workspace without a shell. "
        "Pipes, redirection, shell built-ins, and command chaining are unavailable."
    )

    def __init__(self, config: LocalToolConfig) -> None:
        super().__init__(config)
        self._executables: Dict[str, str] = {}
        for command in config.allowed_commands:
            if Path(command).name != command:
                raise ValueError("allowed_commands entries must be executable names, not paths.")
            # Windows Store app aliases can resolve to a launcher that hangs or
            # opens the Store. Prefer the interpreter already running the agent.
            resolved = sys.executable if command.lower() in {"python", "python.exe"} else shutil.which(command)
            if resolved:
                self._executables[command.lower()] = str(Path(resolved).resolve())

    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "argv": {
                    "type": "array", "items": {"type": "string"},
                    "minItems": 1, "description": "Executable name followed by individual arguments.",
                },
                "cwd": {"type": "string", "description": "Workspace-relative working directory."},
                "timeout_seconds": {"type": "number", "description": "Optional timeout capped by server policy."},
            },
            "required": ["argv"],
        }

    def run(self, argv: List[str], cwd: str = ".", timeout_seconds: float = 0) -> str:
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
            raise RecoverableToolError("argv must be a non-empty array of strings.")
        executable = self._executables.get(argv[0].lower())
        if executable is None:
            available = ", ".join(sorted(self._executables)) or "none"
            raise RecoverableToolError(f"Executable '{argv[0]}' is not allowed or installed. Available: {available}.")
        workdir = self._resolve(cwd, must_exist=True)
        if not workdir.is_dir():
            raise RecoverableToolError(f"cwd is not a directory: {cwd}")
        timeout = min(float(timeout_seconds or self.config.command_timeout_seconds), self.config.command_timeout_seconds)
        if timeout <= 0:
            raise RecoverableToolError("timeout_seconds must be positive.")
        try:
            process = subprocess.Popen(
                [executable, *argv[1:]], cwd=workdir, env=self._safe_environment(),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
            )
            stdout_parts: List[bytes] = []
            stderr_parts: List[bytes] = []
            truncated = {"stdout": False, "stderr": False}

            def drain(stream, parts: List[bytes], key: str) -> None:
                stored = 0
                while True:
                    block = stream.read(8192)
                    if not block:
                        break
                    remaining = self.config.max_command_output_bytes - stored
                    if remaining > 0:
                        parts.append(block[:remaining])
                        stored += min(len(block), remaining)
                    if len(block) > remaining:
                        truncated[key] = True

            readers = [
                threading.Thread(target=drain, args=(process.stdout, stdout_parts, "stdout"), daemon=True),
                threading.Thread(target=drain, args=(process.stderr, stderr_parts, "stderr"), daemon=True),
            ]
            for reader in readers:
                reader.start()
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait(timeout=5)
                for reader in readers:
                    reader.join(timeout=1)
                raise RecoverableToolError(f"Command timed out after {timeout:g} seconds.") from exc
            for reader in readers:
                reader.join(timeout=5)
            stdout = b"".join(stdout_parts)
            stderr = b"".join(stderr_parts)
        except RecoverableToolError:
            raise
        except OSError as exc:
            raise RecoverableToolError(f"Could not start command: {exc}") from exc
        return (
            f"EXIT {exit_code}\nSTDOUT\n{self._decode(stdout, truncated['stdout'])}\n"
            f"STDERR\n{self._decode(stderr, truncated['stderr'])}"
        )

    def _decode(self, output: bytes, truncated: bool) -> str:
        value = output.decode("utf-8", errors="replace")
        return value + ("\n[output truncated]" if truncated else "")

    def _safe_environment(self) -> Dict[str, str]:
        baseline = {
            "PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP",
            "TMPDIR", "LANG", "LC_ALL", "PYTHONIOENCODING",
        }
        allowed = {item.upper() for item in baseline.union(self.config.allowed_environment)}
        result = {key: value for key, value in os.environ.items() if key.upper() in allowed}
        result["PYTHONIOENCODING"] = "utf-8"
        return result


def create_local_tools(
    config: LocalToolConfig, *, include_files: bool = True, include_cli: bool = False
) -> List[BaseTool]:
    tools: List[BaseTool] = []
    if include_files:
        tools.extend([ReadFileTool(config), WriteFileTool(config), ListFilesTool(config)])
    if include_cli:
        tools.append(RunCommandTool(config))
    return tools
