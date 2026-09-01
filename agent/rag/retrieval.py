"""Sparse/dense retrieval and reciprocal-rank fusion."""

from __future__ import annotations

import math
import re
import threading
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from agent.memory.embeddings import EmbeddingProvider
from agent.state.store import BaseVectorStore, NumPyVectorStore

from .models import Chunk, Document, DocumentStatus, MedicalQuery, RetrievalFilters, RetrievalHit
from .repository import RAGRepository


def tokenize(text: str) -> List[str]:
    latin = re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower())
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", text)
    chinese: List[str] = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[index:index + 2] for index in range(len(run) - 1))
    return latin + chinese


class Retriever(Protocol):
    name: str
    def search(self, query: MedicalQuery, limit: int = 20) -> List[RetrievalHit]: ...


def document_matches(document: Document, chunk: Chunk, filters: RetrievalFilters) -> bool:
    return (
        document.status == DocumentStatus.ACTIVE
        and (not filters.jurisdiction or document.jurisdiction == filters.jurisdiction)
        and (not filters.language or document.language == filters.language)
        and (not filters.document_types or document.document_type in filters.document_types)
        and (not filters.publishers or document.publisher in filters.publishers)
        and (not filters.populations or bool(set(filters.populations) & set(chunk.population)))
    )


class BM25Retriever:
    name = "bm25"

    def __init__(self, repository: RAGRepository, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.repository = repository
        self.k1 = k1
        self.b = b
        self._tokens: Dict[str, Counter[str]] = {}
        self._lengths: Dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._lock = threading.RLock()

    def rebuild(self) -> None:
        chunks = self.repository.active_chunks()
        with self._lock:
            self._tokens.clear(); self._lengths.clear(); self._df.clear()
            self._add_chunks(chunks)

    def index_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        with self._lock:
            self._add_chunks(chunk for chunk in chunks if chunk.chunk_type == "child")

    def _add_chunks(self, chunks: Iterable[Chunk]) -> None:
        for chunk in chunks:
            old = self._tokens.get(chunk.id)
            if old:
                self._df.subtract(old.keys())
            counts = Counter(tokenize(chunk.contextual_text))
            self._tokens[chunk.id] = counts
            self._lengths[chunk.id] = sum(counts.values())
            self._df.update(counts.keys())

    def search(self, query: MedicalQuery, limit: int = 20) -> List[RetrievalHit]:
        terms = tokenize(" ".join(query.lexical_queries) or query.normalized)
        with self._lock:
            total = max(1, len(self._tokens))
            average = sum(self._lengths.values()) / total
            scores: Dict[str, float] = defaultdict(float)
            for chunk_id, counts in self._tokens.items():
                chunk = self.repository.get_chunk(chunk_id)
                document = self.repository.get_document(chunk.document_id) if chunk else None
                if not chunk or not document or not document_matches(document, chunk, query.filters):
                    continue
                length = self._lengths[chunk_id]
                for term in terms:
                    frequency = counts.get(term, 0)
                    if not frequency:
                        continue
                    df = max(0, self._df.get(term, 0))
                    idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                    scores[chunk_id] += idf * frequency * (self.k1 + 1) / (
                        frequency + self.k1 * (1 - self.b + self.b * length / max(1, average))
                    )
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [RetrievalHit(item, score, rank + 1, self.name) for rank, (item, score) in enumerate(ranked)]


class DenseRetriever:
    """Embeds chunks and searches them by cosine similarity.

    Vector storage is delegated to a :class:`~agent.state.store.BaseVectorStore`
    (default: in-memory :class:`~agent.state.store.NumPyVectorStore`, matching
    the original hardcoded behavior) so any backend implementing that
    interface -- :class:`~agent.state.store.SQLiteVectorStore` for
    persistence, or a future Chroma/Qdrant/pgvector backend -- can be swapped
    in without touching this class or the callers that construct it.
    """

    name = "dense"

    def __init__(
        self,
        repository: RAGRepository,
        embeddings: EmbeddingProvider,
        vector_store: Optional[BaseVectorStore] = None,
    ) -> None:
        self.repository = repository
        self.embeddings = embeddings
        self._store = vector_store if vector_store is not None else NumPyVectorStore()
        self._lock = threading.RLock()

    def rebuild(self) -> None:
        chunks = self.repository.active_chunks()
        with self._lock:
            self._store.clear()
        self._index(chunks)

    def index_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        self._index([chunk for chunk in chunks if chunk.chunk_type == "child"])

    def _index(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embeddings.embed_documents([chunk.contextual_text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Embedding provider returned the wrong number of vectors.")
        with self._lock:
            for chunk, vector in zip(chunks, vectors):
                # record_id pins the store's id to the chunk's own id (an
                # upsert if this chunk was already indexed) so search() can
                # hand chunk ids straight back to the repository.
                self._store.add(chunk.contextual_text, list(vector), record_id=chunk.id)

    def search(self, query: MedicalQuery, limit: int = 20) -> List[RetrievalHit]:
        vector = self.embeddings.embed_query(" ".join(query.semantic_queries) or query.normalized)
        with self._lock:
            total = len(self._store)
            if not total:
                return []
            # Ask the store for every candidate, not just `limit` -- filtering
            # by document/jurisdiction/population happens *after* the vector
            # search, so under-asking here could drop a match that would
            # have made the cut. A real ANN backend swapped in later would
            # want this over-fetch amount tuned instead of "everything";
            # brute-force backends are the same cost either way.
            hits = self._store.search(vector, k=total)
        scored: List[Tuple[str, float]] = []
        for chunk_id, _text, score in hits:
            chunk = self.repository.get_chunk(chunk_id)
            document = self.repository.get_document(chunk.document_id) if chunk else None
            if chunk and document and document_matches(document, chunk, query.filters):
                scored.append((chunk_id, score))
        # `hits` is already sorted by score descending; filtering preserves
        # that order, so no re-sort is needed.
        return [RetrievalHit(item, score, rank + 1, self.name) for rank, (item, score) in enumerate(scored[:limit])]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    denominator = math.sqrt(sum(x * x for x in left)) * math.sqrt(sum(x * x for x in right))
    return sum(x * y for x, y in zip(left, right)) / denominator if denominator else 0.0


def reciprocal_rank_fusion(
    result_sets: Mapping[str, Sequence[RetrievalHit]], *, rank_constant: int = 60
) -> List[RetrievalHit]:
    scores: Dict[str, float] = defaultdict(float)
    for results in result_sets.values():
        for rank, hit in enumerate(results, 1):
            scores[hit.chunk_id] += 1.0 / (rank_constant + rank)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [RetrievalHit(item, score, rank + 1, "rrf") for rank, (item, score) in enumerate(ranked)]
