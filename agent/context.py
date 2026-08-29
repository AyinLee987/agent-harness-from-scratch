"""Pluggable context providers executed before the first model call."""

from __future__ import annotations

from typing import Any, Dict, Protocol, Sequence


class ContextProvider(Protocol):
    def prepare(self, task: str) -> Sequence[Dict[str, Any]]: ...
