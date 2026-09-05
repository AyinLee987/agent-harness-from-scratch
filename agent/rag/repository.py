"""Authoritative repositories for versioned RAG documents and chunks."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence

from .models import Chunk, Document, DocumentStatus, utc_now


class RAGRepository(Protocol):
    def insert_document(self, document: Document) -> None: ...
    def update_document(self, document: Document) -> None: ...
    def get_document(self, document_id: str) -> Optional[Document]: ...
    def find_by_checksum(self, checksum: str) -> Optional[Document]: ...
    def list_documents(self, status: Optional[DocumentStatus] = None) -> List[Document]: ...
    def insert_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]: ...
    def get_chunks(self, chunk_ids: Sequence[str]) -> List[Chunk]: ...
    def chunks_for_document(self, document_id: str) -> List[Chunk]: ...
    def active_chunks(self) -> List[Chunk]: ...
    def publish(self, document_id: str) -> Document: ...


class InMemoryRAGRepository:
    def __init__(self) -> None:
        self._documents: Dict[str, Document] = {}
        self._chunks: Dict[str, Chunk] = {}
        self._lock = threading.RLock()

    def insert_document(self, document: Document) -> None:
        with self._lock:
            if document.id in self._documents:
                raise ValueError(f"Duplicate document id {document.id!r}.")
            self._documents[document.id] = copy.deepcopy(document)

    def update_document(self, document: Document) -> None:
        with self._lock:
            if document.id not in self._documents:
                raise KeyError(document.id)
            self._documents[document.id] = copy.deepcopy(document)

    def get_document(self, document_id: str) -> Optional[Document]:
        with self._lock:
            item = self._documents.get(document_id)
            return copy.deepcopy(item) if item else None

    def find_by_checksum(self, checksum: str) -> Optional[Document]:
        """The live document with this content, else the most recent one.

        Once identical content can exist at more than one version (see
        ``RAGIngestionService`` and BUGS.md #20), "any row with this
        checksum" is ambiguous. ACTIVE wins, because a caller asking
        whether this content is already indexed is asking about the
        effective version.
        """

        with self._lock:
            matches = [
                item
                for item in self._documents.values()
                if item.checksum == checksum and item.status != DocumentStatus.FAILED
            ]
        if not matches:
            return None
        active = [item for item in matches if item.status == DocumentStatus.ACTIVE]
        return copy.deepcopy((active or matches)[-1])

    def list_documents(self, status: Optional[DocumentStatus] = None) -> List[Document]:
        with self._lock:
            items = list(self._documents.values())
            if status is not None:
                items = [item for item in items if item.status == status]
            return copy.deepcopy(items)

    def insert_chunks(self, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            for chunk in chunks:
                if chunk.id in self._chunks:
                    raise ValueError(f"Duplicate chunk id {chunk.id!r}.")
            for chunk in chunks:
                self._chunks[chunk.id] = copy.deepcopy(chunk)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            item = self._chunks.get(chunk_id)
            return copy.deepcopy(item) if item else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> List[Chunk]:
        with self._lock:
            return [copy.deepcopy(self._chunks[item]) for item in chunk_ids if item in self._chunks]

    def chunks_for_document(self, document_id: str) -> List[Chunk]:
        with self._lock:
            items = [item for item in self._chunks.values() if item.document_id == document_id]
            items.sort(key=lambda item: item.sequence)
            return copy.deepcopy(items)

    def active_chunks(self) -> List[Chunk]:
        with self._lock:
            active_ids = {
                item.id for item in self._documents.values() if item.status == DocumentStatus.ACTIVE
            }
            return copy.deepcopy(
                [item for item in self._chunks.values() if item.document_id in active_ids and item.chunk_type == "child"]
            )

    def publish(self, document_id: str) -> Document:
        with self._lock:
            current = self._documents.get(document_id)
            if current is None:
                raise KeyError(document_id)
            if current.status != DocumentStatus.STAGING:
                raise ValueError("Only staging documents can be published.")
            if not any(item.document_id == document_id for item in self._chunks.values()):
                raise ValueError("Cannot publish a document without chunks.")
            for item in self._documents.values():
                if item.logical_id == current.logical_id and item.status == DocumentStatus.ACTIVE:
                    item.status = DocumentStatus.SUPERSEDED
                    item.superseded_by_id = current.id
                    current.supersedes_id = item.id
            current.status = DocumentStatus.ACTIVE
            return copy.deepcopy(current)


def content_checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SQLiteRAGRepository:
    """SQLite source of truth with an atomic document publish operation."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    id TEXT PRIMARY KEY, logical_id TEXT NOT NULL, checksum TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_rag_documents_logical_status
                    ON rag_documents(logical_id, status);
                CREATE INDEX IF NOT EXISTS idx_rag_documents_checksum
                    ON rag_documents(checksum);
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    chunk_type TEXT NOT NULL, sequence_no INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    FOREIGN KEY(document_id) REFERENCES rag_documents(id)
                );
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document
                    ON rag_chunks(document_id, sequence_no);
                """
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def insert_document(self, document: Document) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO rag_documents VALUES (?, ?, ?, ?, ?)", self._document_values(document)
            )

    def update_document(self, document: Document) -> None:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "UPDATE rag_documents SET logical_id=?, checksum=?, status=?, payload=? WHERE id=?",
                (*self._document_values(document)[1:], document.id),
            )
            if not cursor.rowcount:
                raise KeyError(document.id)

    def get_document(self, document_id: str) -> Optional[Document]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM rag_documents WHERE id=?", (document_id,)
            ).fetchone()
        return self._decode_document(row["payload"]) if row else None

    def find_by_checksum(self, checksum: str) -> Optional[Document]:
        """The live document with this content, else the most recent one.

        See :meth:`InMemoryRAGRepository.find_by_checksum` -- the two must
        agree, and ``ORDER BY rowid DESC`` alone did not prefer ACTIVE.
        """

        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM rag_documents WHERE checksum=? AND status!=? "
                "ORDER BY (status=?) DESC, rowid DESC LIMIT 1",
                (
                    checksum,
                    DocumentStatus.FAILED.value,
                    DocumentStatus.ACTIVE.value,
                ),
            ).fetchone()
        return self._decode_document(row["payload"]) if row else None

    def list_documents(self, status: Optional[DocumentStatus] = None) -> List[Document]:
        sql = "SELECT payload FROM rag_documents"
        parameters: tuple = ()
        if status is not None:
            sql += " WHERE status=?"
            parameters = (status.value,)
        sql += " ORDER BY rowid"
        with self._lock:
            rows = self._connection.execute(sql, parameters).fetchall()
        return [self._decode_document(row["payload"]) for row in rows]

    def insert_chunks(self, chunks: Sequence[Chunk]) -> None:
        values = [
            (chunk.id, chunk.document_id, chunk.chunk_type, chunk.sequence, self._json(chunk))
            for chunk in chunks
        ]
        with self._lock, self._connection:
            self._connection.executemany("INSERT INTO rag_chunks VALUES (?, ?, ?, ?, ?)", values)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM rag_chunks WHERE id=?", (chunk_id,)
            ).fetchone()
        return self._decode_chunk(row["payload"]) if row else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> List[Chunk]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT id, payload FROM rag_chunks WHERE id IN ({placeholders})", tuple(chunk_ids)
            ).fetchall()
        found = {row["id"]: self._decode_chunk(row["payload"]) for row in rows}
        return [found[item] for item in chunk_ids if item in found]

    def chunks_for_document(self, document_id: str) -> List[Chunk]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM rag_chunks WHERE document_id=? ORDER BY sequence_no", (document_id,)
            ).fetchall()
        return [self._decode_chunk(row["payload"]) for row in rows]

    def active_chunks(self) -> List[Chunk]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT c.payload FROM rag_chunks c JOIN rag_documents d ON c.document_id=d.id "
                "WHERE d.status=? AND c.chunk_type='child' ORDER BY c.sequence_no",
                (DocumentStatus.ACTIVE.value,),
            ).fetchall()
        return [self._decode_chunk(row["payload"]) for row in rows]

    def publish(self, document_id: str) -> Document:
        with self._lock, self._connection:
            current = self.get_document(document_id)
            if current is None:
                raise KeyError(document_id)
            if current.status != DocumentStatus.STAGING:
                raise ValueError("Only staging documents can be published.")
            count = self._connection.execute(
                "SELECT COUNT(*) FROM rag_chunks WHERE document_id=?", (document_id,)
            ).fetchone()[0]
            if not count:
                raise ValueError("Cannot publish a document without chunks.")
            rows = self._connection.execute(
                "SELECT payload FROM rag_documents WHERE logical_id=? AND status=? AND id!=?",
                (current.logical_id, DocumentStatus.ACTIVE.value, current.id),
            ).fetchall()
            for row in rows:
                previous = self._decode_document(row["payload"])
                previous.status = DocumentStatus.SUPERSEDED
                previous.superseded_by_id = current.id
                current.supersedes_id = previous.id
                self._update_document_no_commit(previous)
            current.status = DocumentStatus.ACTIVE
            self._update_document_no_commit(current)
        return current

    def _update_document_no_commit(self, document: Document) -> None:
        self._connection.execute(
            "UPDATE rag_documents SET logical_id=?, checksum=?, status=?, payload=? WHERE id=?",
            (*self._document_values(document)[1:], document.id),
        )

    @classmethod
    def _json(cls, value) -> str:
        def default(item):
            if isinstance(item, datetime):
                return {"__datetime__": item.isoformat()}
            if isinstance(item, Enum):
                return item.value
            raise TypeError(type(item).__name__)
        return json.dumps(dataclasses.asdict(value), ensure_ascii=False, default=default)

    @classmethod
    def _loads(cls, payload: str) -> dict:
        def hook(item):
            if set(item) == {"__datetime__"}:
                return datetime.fromisoformat(item["__datetime__"])
            return item
        return json.loads(payload, object_hook=hook)

    @classmethod
    def _decode_document(cls, payload: str) -> Document:
        data = cls._loads(payload)
        data["status"] = DocumentStatus(data["status"])
        return Document(**data)

    @classmethod
    def _decode_chunk(cls, payload: str) -> Chunk:
        return Chunk(**cls._loads(payload))

    @classmethod
    def _document_values(cls, document: Document) -> tuple:
        return document.id, document.logical_id, document.checksum, document.status.value, cls._json(document)
