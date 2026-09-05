"""Durable storage for suspended agent runs.

A run that suspends on a long-running job (see :mod:`agent.jobs`) has to
put its state *somewhere* that outlives the request that started it --
otherwise "suspend and resume" is only true within one process's lifetime,
which is exactly the guarantee a half-hour job needs and the one a heap
dict cannot give.

This belongs to the State layer rather than to :mod:`agent.jobs`: what is
stored is an :class:`~agent.state.context.ExecutionContext` snapshot, i.e.
what the agent *knows*, and jobs are only today's reason for wanting it
persisted. A future resumable-on-approval or resumable-on-webhook flow
would store the same thing.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class RunCheckpoint:
    """One suspended run, ready to be picked back up."""

    run_id: str
    task: str
    checkpoint: Dict[str, Any]
    pending_job_ids: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_record(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task": self.task,
            "checkpoint": self.checkpoint,
            "pending_job_ids": self.pending_job_ids,
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "RunCheckpoint":
        return cls(
            run_id=str(record["run_id"]),
            task=str(record.get("task") or ""),
            checkpoint=dict(record.get("checkpoint") or {}),
            pending_job_ids=list(record.get("pending_job_ids") or []),
            conversation_id=record.get("conversation_id"),
            created_at=float(record.get("created_at") or time.time()),
            updated_at=float(record.get("updated_at") or time.time()),
        )


class CheckpointStore(Protocol):
    def save(self, checkpoint: RunCheckpoint) -> None: ...

    def load(self, run_id: str) -> Optional[RunCheckpoint]: ...

    def delete(self, run_id: str) -> bool: ...

    def list_suspended(self) -> List[RunCheckpoint]: ...


class InMemoryCheckpointStore:
    """Zero-dependency reference implementation, and what tests use."""

    def __init__(self) -> None:
        self._runs: Dict[str, RunCheckpoint] = {}
        self._lock = threading.RLock()

    def save(self, checkpoint: RunCheckpoint) -> None:
        with self._lock:
            checkpoint.updated_at = time.time()
            self._runs[checkpoint.run_id] = copy.deepcopy(checkpoint)

    def load(self, run_id: str) -> Optional[RunCheckpoint]:
        with self._lock:
            item = self._runs.get(run_id)
            return copy.deepcopy(item) if item is not None else None

    def delete(self, run_id: str) -> bool:
        with self._lock:
            return self._runs.pop(run_id, None) is not None

    def list_suspended(self) -> List[RunCheckpoint]:
        with self._lock:
            return copy.deepcopy(
                sorted(self._runs.values(), key=lambda item: item.created_at)
            )


class SQLiteCheckpointStore:
    """SQLite source of truth using a version-tolerant JSON payload."""

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
                CREATE TABLE IF NOT EXISTS run_checkpoints (
                    run_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    checkpoint_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_created "
                "ON run_checkpoints(created_at)"
            )
            self._conn.commit()

    def save(self, checkpoint: RunCheckpoint) -> None:
        checkpoint.updated_at = time.time()
        payload = json.dumps(checkpoint.to_record(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO run_checkpoints VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET "
                "conversation_id=excluded.conversation_id, "
                "updated_at=excluded.updated_at, "
                "checkpoint_json=excluded.checkpoint_json",
                (
                    checkpoint.run_id,
                    checkpoint.conversation_id,
                    checkpoint.created_at,
                    checkpoint.updated_at,
                    payload,
                ),
            )
            self._conn.commit()

    def load(self, run_id: str) -> Optional[RunCheckpoint]:
        with self._lock:
            row = self._conn.execute(
                "SELECT checkpoint_json FROM run_checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return RunCheckpoint.from_record(json.loads(row["checkpoint_json"])) if row else None

    def delete(self, run_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM run_checkpoints WHERE run_id = ?", (run_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def list_suspended(self) -> List[RunCheckpoint]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT checkpoint_json FROM run_checkpoints ORDER BY created_at"
            ).fetchall()
        return [
            RunCheckpoint.from_record(json.loads(row["checkpoint_json"])) for row in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
