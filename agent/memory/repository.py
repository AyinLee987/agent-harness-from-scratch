"""Authoritative memory repositories; vector indexes are derived data."""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .models import (
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    RetentionPolicy,
    Sensitivity,
)


class MemoryRepository(Protocol):
    def insert(self, record: MemoryRecord) -> None:
        ...

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        ...

    def update(self, record: MemoryRecord) -> None:
        ...

    def list_records(
        self,
        *,
        namespace: Optional[str] = None,
        subject_id: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
    ) -> List[MemoryRecord]:
        ...

    def find_active_by_hash(self, content_hash: str) -> Optional[MemoryRecord]:
        ...

    def delete_physical(self, record_id: str) -> bool:
        ...


class InMemoryMemoryRepository:
    def __init__(self) -> None:
        self._records: Dict[str, MemoryRecord] = {}
        self._lock = threading.RLock()

    def insert(self, record: MemoryRecord) -> None:
        with self._lock:
            if record.id in self._records:
                raise ValueError(f"Duplicate memory id {record.id!r}.")
            self._records[record.id] = copy.deepcopy(record)

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            record = self._records.get(record_id)
            return copy.deepcopy(record) if record is not None else None

    def update(self, record: MemoryRecord) -> None:
        with self._lock:
            if record.id not in self._records:
                raise KeyError(record.id)
            self._records[record.id] = copy.deepcopy(record)

    def list_records(
        self,
        *,
        namespace: Optional[str] = None,
        subject_id: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
    ) -> List[MemoryRecord]:
        with self._lock:
            records = list(self._records.values())
            if namespace is not None:
                records = [item for item in records if item.namespace == namespace]
            if subject_id is not None:
                records = [item for item in records if item.subject_id == subject_id]
            if status is not None:
                records = [item for item in records if item.status == status]
            records.sort(key=lambda item: item.created_at, reverse=True)
            return copy.deepcopy(records)

    def find_active_by_hash(self, content_hash: str) -> Optional[MemoryRecord]:
        with self._lock:
            for record in self._records.values():
                if record.content_hash == content_hash and record.status == MemoryStatus.ACTIVE:
                    return copy.deepcopy(record)
        return None

    def delete_physical(self, record_id: str) -> bool:
        with self._lock:
            return self._records.pop(record_id, None) is not None


class SQLiteMemoryRepository:
    """SQLite source of truth using a version-tolerant JSON record payload."""

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
                CREATE TABLE IF NOT EXISTS memory_records (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL,
                    subject_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    record_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_scope "
                "ON memory_records(namespace, subject_id, status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_hash "
                "ON memory_records(content_hash, status)"
            )
            self._conn.commit()

    def insert(self, record: MemoryRecord) -> None:
        values = self._values(record)
        with self._lock:
            self._conn.execute(
                "INSERT INTO memory_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                values,
            )
            self._conn.commit()

    def get(self, record_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT record_json FROM memory_records WHERE id = ?", (record_id,)
            ).fetchone()
        return self._decode(row["record_json"]) if row else None

    def update(self, record: MemoryRecord) -> None:
        values = self._values(record)
        with self._lock:
            cursor = self._conn.execute(
                """
                UPDATE memory_records SET
                    namespace=?, subject_id=?, kind=?, status=?, content_hash=?,
                    created_at=?, updated_at=?, record_json=?
                WHERE id=?
                """,
                values[1:] + values[:1],
            )
            if cursor.rowcount == 0:
                raise KeyError(record.id)
            self._conn.commit()

    def list_records(
        self,
        *,
        namespace: Optional[str] = None,
        subject_id: Optional[str] = None,
        status: Optional[MemoryStatus] = None,
    ) -> List[MemoryRecord]:
        clauses: List[str] = []
        params: List[str] = []
        if namespace is not None:
            clauses.append("namespace = ?")
            params.append(namespace)
        if subject_id is not None:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT record_json FROM memory_records{where} ORDER BY created_at DESC",
                params,
            ).fetchall()
        return [self._decode(row["record_json"]) for row in rows]

    def find_active_by_hash(self, content_hash: str) -> Optional[MemoryRecord]:
        with self._lock:
            row = self._conn.execute(
                "SELECT record_json FROM memory_records "
                "WHERE content_hash = ? AND status = ? LIMIT 1",
                (content_hash, MemoryStatus.ACTIVE.value),
            ).fetchone()
        return self._decode(row["record_json"]) if row else None

    def delete_physical(self, record_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memory_records WHERE id = ?", (record_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _values(record: MemoryRecord) -> tuple:
        return (
            record.id,
            record.namespace,
            record.subject_id,
            record.kind.value,
            record.status.value,
            record.content_hash,
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
            json.dumps(_record_to_dict(record), ensure_ascii=False),
        )

    @staticmethod
    def _decode(payload: str) -> MemoryRecord:
        data = json.loads(payload)
        for name in (
            "created_at",
            "updated_at",
            "last_accessed_at",
            "expires_at",
            "review_at",
            "deleted_at",
        ):
            if data.get(name):
                data[name] = datetime.fromisoformat(data[name])
        data["kind"] = MemoryKind(data["kind"])
        data["status"] = MemoryStatus(data["status"])
        data["retention_policy"] = RetentionPolicy(data["retention_policy"])
        data["sensitivity"] = Sensitivity(data["sensitivity"])
        return MemoryRecord(**data)


def _record_to_dict(record: MemoryRecord) -> dict:
    data = asdict(record)
    for name in ("kind", "status", "retention_policy", "sensitivity"):
        data[name] = getattr(record, name).value
    for name in (
        "created_at",
        "updated_at",
        "last_accessed_at",
        "expires_at",
        "review_at",
        "deleted_at",
    ):
        value = getattr(record, name)
        data[name] = value.isoformat() if value is not None else None
    return data
