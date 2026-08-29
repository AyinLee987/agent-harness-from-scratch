"""Replaceable vector indexes containing only derived memory data."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Set

import numpy as np

from .errors import MemoryConfigurationError
from .models import MemoryKind


@dataclass
class VectorHit:
    record_id: str
    score: float


class MemoryVectorIndex(Protocol):
    def upsert(
        self,
        record_id: str,
        vector: Sequence[float],
        *,
        namespace: str,
        subject_id: str,
        kind: MemoryKind,
    ) -> None:
        ...

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        namespace: Optional[str] = None,
        subject_id: Optional[str] = None,
        kinds: Optional[Set[MemoryKind]] = None,
    ) -> List[VectorHit]:
        ...

    def delete(self, record_id: str) -> bool:
        ...


class InMemoryVectorIndex:
    """Thread-safe exact vector index for tests and small deployments."""

    def __init__(self) -> None:
        self._items: Dict[str, dict] = {}
        self._dimension: Optional[int] = None
        self._lock = threading.RLock()

    def upsert(
        self,
        record_id: str,
        vector: Sequence[float],
        *,
        namespace: str,
        subject_id: str,
        kind: MemoryKind,
    ) -> None:
        array = np.asarray(vector, dtype=np.float32)
        self._validate_dimension(array)
        with self._lock:
            self._items[record_id] = {
                "vector": array,
                "namespace": namespace,
                "subject_id": subject_id,
                "kind": kind,
            }

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int,
        namespace: Optional[str] = None,
        subject_id: Optional[str] = None,
        kinds: Optional[Set[MemoryKind]] = None,
    ) -> List[VectorHit]:
        query = np.asarray(vector, dtype=np.float32)
        self._validate_dimension(query)
        scored: List[VectorHit] = []
        with self._lock:
            for record_id, item in self._items.items():
                if namespace is not None and item["namespace"] != namespace:
                    continue
                if subject_id is not None and item["subject_id"] != subject_id:
                    continue
                if kinds is not None and item["kind"] not in kinds:
                    continue
                scored.append(VectorHit(record_id, _cosine(query, item["vector"])))
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[: max(0, limit)]

    def delete(self, record_id: str) -> bool:
        with self._lock:
            return self._items.pop(record_id, None) is not None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _validate_dimension(self, vector: np.ndarray) -> None:
        if vector.ndim != 1 or len(vector) == 0:
            raise MemoryConfigurationError("Memory embeddings must be non-empty vectors.")
        with self._lock:
            if self._dimension is None:
                self._dimension = len(vector)
            elif len(vector) != self._dimension:
                raise MemoryConfigurationError(
                    f"Vector index expects {self._dimension} dimensions, got {len(vector)}."
                )


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left)) * float(np.linalg.norm(right))
    return float(np.dot(left, right)) / denominator if denominator else 0.0
