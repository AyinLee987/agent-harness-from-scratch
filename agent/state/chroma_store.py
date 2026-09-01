"""Chroma-backed vector store -- an optional-dependency BaseVectorStore.

Requires the ``chromadb`` package (``pip install chromadb``). Nothing else
in ``agent`` imports ``chromadb`` at module load time -- the import is
deferred to :meth:`ChromaVectorStore.__init__`, the same pattern
``OpenAICompatibleEmbeddingProvider`` uses for the optional ``openai``
package -- so the rest of the library, and every other ``BaseVectorStore``
backend, keeps working with zero extra dependencies if this class is never
instantiated.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .store import BaseVectorStore


class VectorStoreConfigurationError(RuntimeError):
    """Raised when a vector store backend can't be constructed as configured."""


class ChromaVectorStore(BaseVectorStore):
    """Persistent-or-ephemeral vector store backed by Chroma.

    .. code-block:: python

        from agent.state import ChromaVectorStore, LongTermMemory

        # Persistent, on disk:
        store = ChromaVectorStore(persist_directory="memory/chroma")
        mem = LongTermMemory(llm, vector_store=store)

        # Or drop it straight into RAG's dense retriever:
        from agent.rag import DenseRetriever
        dense = DenseRetriever(repository, embeddings, vector_store=store)

    Same ``add``/``search``/``delete``/``clear``/``all`` contract as
    :class:`~agent.state.store.NumPyVectorStore` and
    :class:`~agent.state.store.SQLiteVectorStore` -- exercised by the same
    parametrized suite in ``tests/test_vector_store.py`` (add ``"chroma"``
    to that fixture's params to run it here too).

    Two real-Chroma details this class works around to honor that shared
    contract:

    * Chroma computes *distance*, not similarity, and defaults to squared
      L2 -- the collection is created with ``hnsw:space: cosine`` so
      ``score = 1 - distance`` matches the cosine-similarity every other
      backend returns.
    * Chroma's own ``upsert`` *merges* metadata into an existing id instead
      of replacing it outright. :meth:`add` deletes the id first (a no-op
      if it's new) so calling it twice with the same ``record_id`` fully
      replaces the record, matching the upsert semantics
      :class:`~agent.state.store.BaseVectorStore` documents.

    One real Chroma constraint this class does *not* work around: metadata
    values must be ``str``/``int``/``float``/``bool`` (no ``None``, no
    nested dict/list) -- Chroma itself will raise if you pass something
    else in ``metadata``.
    """

    def __init__(
        self,
        *,
        persist_directory: Optional[str] = None,
        collection_name: str = "agent_vectors",
        client: Any = None,
    ) -> None:
        if client is None:
            try:
                import chromadb
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise VectorStoreConfigurationError(
                    "ChromaVectorStore requires the 'chromadb' package "
                    "(pip install chromadb)."
                ) from exc
            client = (
                chromadb.PersistentClient(path=persist_directory)
                if persist_directory
                else chromadb.EphemeralClient()
            )
        self._client = client
        # embedding_function=None: embeddings always arrive pre-computed
        # from an EmbeddingProvider upstream (LongTermMemory/DenseRetriever
        # both embed before calling add()/search()) -- Chroma must never
        # compute its own.
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    # -- BaseVectorStore interface ------------------------------------------
    def add(
        self,
        text: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        record_id: Optional[str] = None,
    ) -> str:
        record_id = record_id or uuid.uuid4().hex[:12]
        self._collection.delete(ids=[record_id])  # see class docstring
        payload = dict(metadata or {})
        payload["_created_at"] = datetime.now(timezone.utc).isoformat()
        self._collection.upsert(
            ids=[record_id],
            embeddings=[list(embedding)],
            documents=[text],
            metadatas=[payload],
        )
        return record_id

    def search(
        self, query_embedding: List[float], k: int = 3
    ) -> List[Tuple[str, str, float]]:
        n = min(k, self._collection.count())
        if n <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[list(query_embedding)], n_results=n
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        distances = result["distances"][0]
        return [
            (record_id, document or "", 1.0 - distance)
            for record_id, document, distance in zip(ids, documents, distances)
        ]

    def delete(self, record_id: str) -> bool:
        existing = self._collection.get(ids=[record_id])
        if not existing["ids"]:
            return False
        self._collection.delete(ids=[record_id])
        return True

    def clear(self) -> None:
        existing_ids = self._collection.get()["ids"]
        if existing_ids:
            self._collection.delete(ids=existing_ids)

    def all(self) -> List[Dict[str, Any]]:
        data = self._collection.get()
        ids = data["ids"]
        documents = data.get("documents") or [None] * len(ids)
        metadatas = data.get("metadatas") or [None] * len(ids)
        records: List[Dict[str, Any]] = []
        for record_id, document, meta in zip(ids, documents, metadatas):
            meta = dict(meta or {})
            created_at = meta.pop("_created_at", "")
            records.append({
                "id": record_id,
                "text": document or "",
                "metadata": meta,
                "created_at": created_at,
            })
        return records

    def __len__(self) -> int:
        return self._collection.count()
