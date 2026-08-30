"""Conversation-level message storage separate from a single run context."""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
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


class SQLiteSessionStore:
    """SQLite-backed :class:`SessionMemoryStore` — the durable counterpart to
    :class:`InMemorySessionStore`.

    Mirrors :class:`~agent.memory.repository.SQLiteMemoryRepository`'s
    version-tolerant approach: each message is stored as an opaque JSON blob
    keyed by an autoincrementing sequence number, so a message dict's shape
    (extra keys like ``tool_calls``) never needs a schema migration.
    """

    def __init__(self, db_path: str) -> None:
        if db_path == ":memory:":
            self.db_path = db_path
        else:
            path = Path(db_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_conversation "
                "ON session_messages(conversation_id, seq)"
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    through_message_id TEXT,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.commit()

    def load_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT message_json FROM session_messages "
                "WHERE conversation_id = ? ORDER BY seq",
                (conversation_id,),
            ).fetchall()
        return [json.loads(row["message_json"]) for row in rows]

    def append_message(self, conversation_id: str, message: Dict[str, Any]) -> str:
        message_id = str(message.get("id") or uuid.uuid4().hex[:12])
        stored = {**message, "id": message_id}
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_messages (conversation_id, message_id, message_json) "
                "VALUES (?, ?, ?)",
                (conversation_id, message_id, json.dumps(stored, ensure_ascii=False)),
            )
            self._conn.commit()
        return message_id

    def load_summary(self, conversation_id: str) -> Optional[SummarySnapshot]:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary, through_message_id, updated_at FROM session_summaries "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        return SummarySnapshot(
            conversation_id=conversation_id,
            summary=row["summary"],
            through_message_id=row["through_message_id"],
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def save_summary(self, snapshot: SummarySnapshot) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO session_summaries
                    (conversation_id, summary, through_message_id, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary=excluded.summary,
                    through_message_id=excluded.through_message_id,
                    updated_at=excluded.updated_at
                """,
                (
                    snapshot.conversation_id,
                    snapshot.summary,
                    snapshot.through_message_id,
                    snapshot.updated_at.isoformat(),
                ),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
