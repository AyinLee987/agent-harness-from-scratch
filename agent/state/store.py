"""Pluggable vector store backends for long-term memory.

This module provides a clean abstraction over vector storage so
:class:`~agent.memory.LongTermMemory` can swap backends without changing
its public interface.

Available backends:

* :class:`NumPyVectorStore` — in-memory NumPy array (zero deps, default).
* :class:`SQLiteVectorStore` — persistent SQLite-backed storage (zero extra deps,
  uses Python's built-in ``sqlite3``).

All backends implement :class:`BaseVectorStore` so you can drop in a new one
(FAISS, Qdrant, pgvector, etc.) behind the same ``add``/``search``/``delete``
contract.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseVectorStore(ABC):
    """Abstract interface for a vector store.

    Every backend must implement :meth:`add`, :meth:`search`, :meth:`delete`,
    :meth:`all`, and ``__len__``.
    """

    @abstractmethod
    def add(
        self, text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store a text + embedding pair. Returns a unique record id."""

    @abstractmethod
    def search(
        self, query_embedding: List[float], k: int = 3
    ) -> List[Tuple[str, str, float]]:
        """Return up to ``k`` ``(id, text, score)`` tuples ranked by cosine similarity."""

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Remove a record by id. Return ``True`` if it existed."""

    @abstractmethod
    def all(self) -> List[Dict[str, Any]]:
        """Return all records as dicts (without embedding vectors, for listing)."""

    @abstractmethod
    def __len__(self) -> int:
        """Number of stored records."""


# ---------------------------------------------------------------------------
# In-memory NumPy backend (the original backed, now behind the interface)
# ---------------------------------------------------------------------------
@np.errstate(invalid="ignore")
def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D float arrays."""
    dot = float(np.dot(a, b))
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    return dot / denom if denom > 0 else 0.0


class NumPyVectorStore(BaseVectorStore):
    """In-memory vector store using NumPy arrays.

    The original backend — no persistence, no external dependencies beyond
    NumPy.  Good for testing, demos, and tiny workloads.
    """

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    # -- BaseVectorStore interface ------------------------------------------
    def add(
        self, text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        record_id = uuid.uuid4().hex[:12]
        self._records.append(
            {
                "id": record_id,
                "text": text,
                "embedding": np.asarray(embedding, dtype=np.float32),
                "metadata": metadata or {},
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        return record_id

    def search(
        self, query_embedding: List[float], k: int = 3
    ) -> List[Tuple[str, str, float]]:
        if not self._records:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        scored: List[Tuple[str, str, float]] = []
        for rec in self._records:
            score = _cosine_similarity(q, rec["embedding"])
            scored.append((rec["id"], rec["text"], score))
        scored.sort(key=lambda pair: pair[2], reverse=True)
        return scored[:k]

    def delete(self, record_id: str) -> bool:
        before = len(self._records)
        self._records = [r for r in self._records if r["id"] != record_id]
        return len(self._records) < before

    def all(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "metadata": r["metadata"],
                "created_at": r["created_at"],
            }
            for r in self._records
        ]

    def __len__(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Persistent SQLite backend
# ---------------------------------------------------------------------------
class SQLiteVectorStore(BaseVectorStore):
    """Persistent vector store backed by a SQLite database.

    Embeddings are stored as binary BLOBs (packed float32 arrays).  Similarity
    search loads all vectors into memory and computes cosine similarity in
    Python — simple and fine for up to ~100k records.

    Args:
        db_path: Path to the SQLite file.  Defaults to ``memory/vector_store.db``.
            Set to ``:memory:`` for an in-memory database (useful for testing).

    .. code-block:: python

        store = SQLiteVectorStore("memory/knowledge.db")
        store.add("Paris is the capital of France.", embedding)
        results = store.search(query_embedding, k=3)
        for id_, text, score in results:
            print(f"[{score:.3f}] {text}")
    """

    def __init__(self, db_path: str | None = None) -> None:
        if db_path is None:
            db_dir = os.path.join(os.getcwd(), "memory")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "vector_store.db")
        else:
            # Ensure the parent directory exists.
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        self.db_path = os.path.abspath(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

    def _create_table(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id          TEXT PRIMARY KEY,
                text        TEXT NOT NULL,
                embedding   BLOB NOT NULL,
                metadata    TEXT DEFAULT '{}',
                created_at  TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vectors_created ON vectors(created_at)"
        )
        self._conn.commit()

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _pack(embedding: List[float]) -> bytes:
        """Pack a list of floats into a binary BLOB (float32, little-endian)."""
        return struct.pack(f"<{len(embedding)}f", *embedding)

    @staticmethod
    def _unpack(blob: bytes) -> List[float]:
        """Unpack a binary BLOB back into a list of floats."""
        count = len(blob) // 4
        return list(struct.unpack(f"<{count}f", blob))

    # -- BaseVectorStore interface ------------------------------------------
    def add(
        self, text: str, embedding: List[float], metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        record_id = uuid.uuid4().hex[:12]
        self._conn.execute(
            """
            INSERT INTO vectors (id, text, embedding, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                record_id,
                text,
                self._pack(embedding),
                json.dumps(metadata or {}, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return record_id

    def search(
        self, query_embedding: List[float], k: int = 3
    ) -> List[Tuple[str, str, float]]:
        rows = self._conn.execute(
            "SELECT id, text, embedding FROM vectors"
        ).fetchall()
        if not rows:
            return []

        q = np.asarray(query_embedding, dtype=np.float32)
        scored: List[Tuple[str, str, float]] = []
        for row in rows:
            vec = np.asarray(self._unpack(row["embedding"]), dtype=np.float32)
            score = _cosine_similarity(q, vec)
            scored.append((row["id"], row["text"], score))

        scored.sort(key=lambda pair: pair[2], reverse=True)
        return scored[:k]

    def delete(self, record_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM vectors WHERE id = ?", (record_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def all(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT id, text, metadata, created_at FROM vectors ORDER BY created_at DESC"
        ).fetchall()
        return [
            {
                "id": row["id"],
                "text": row["text"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def __len__(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM vectors").fetchone()
        return row["cnt"] if row else 0

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
