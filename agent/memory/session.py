"""Conversation-level message storage separate from a single run context."""

from __future__ import annotations

import copy
import threading
import uuid
from typing import Any, Dict, List, Optional, Protocol

from .models import SummarySnapshot


class SessionMemoryStore(Protocol):
    def load_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        ...

    def append_message(
        self, conversation_id: str, message: Dict[str, Any]
    ) -> str:
        ...

    def load_summary(self, conversation_id: str) -> Optional[SummarySnapshot]:
        ...

    def save_summary(self, snapshot: SummarySnapshot) -> None:
        ...


class InMemorySessionStore:
    """Thread-safe session store for tests; production can replace this backend."""

    def __init__(self) -> None:
        self._messages: Dict[str, List[Dict[str, Any]]] = {}
        self._summaries: Dict[str, SummarySnapshot] = {}
        self._lock = threading.RLock()

    def load_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return copy.deepcopy(self._messages.get(conversation_id, []))

    def append_message(
        self, conversation_id: str, message: Dict[str, Any]
    ) -> str:
        message_id = str(message.get("id") or uuid.uuid4().hex[:12])
        stored = {**message, "id": message_id}
        with self._lock:
            self._messages.setdefault(conversation_id, []).append(copy.deepcopy(stored))
        return message_id

    def load_summary(self, conversation_id: str) -> Optional[SummarySnapshot]:
        with self._lock:
            snapshot = self._summaries.get(conversation_id)
            return copy.deepcopy(snapshot) if snapshot is not None else None

    def save_summary(self, snapshot: SummarySnapshot) -> None:
        with self._lock:
            self._summaries[snapshot.conversation_id] = copy.deepcopy(snapshot)
