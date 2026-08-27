"""Helpers for launching MCP server packages without debugger interference."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def uv_tool_command(
    package: str,
    *,
    python_executable: Optional[str] = None,
    path_lookup: Callable[[str], Optional[str]] = shutil.which,
) -> Tuple[str, List[str]]:
    """Return the native uv executable and arguments for an isolated tool run.

    Launching uv through Python causes debuggers that trace subprocesses to
    attach to uv and potentially corrupt an MCP server's stdio channel. Calling
    uv's native executable avoids that instrumentation.
    """

    python_path = Path(python_executable or sys.executable)
    executable_name = "uv.exe" if os.name == "nt" else "uv"
    candidates = [
        python_path.parent / executable_name,
        python_path.parent / "Scripts" / executable_name,
    ]
    discovered = path_lookup("uv")
    if discovered:
        candidates.append(Path(discovered))

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate), ["tool", "run", package]
    raise FileNotFoundError(
        "uv executable not found; install it with: python -m pip install uv"
    )
