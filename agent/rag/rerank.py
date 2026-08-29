"""Replaceable reranking layer."""

from __future__ import annotations

from typing import Callable, List, Protocol, Sequence

from .models import Evidence, MedicalQuery
from .retrieval import tokenize


class Reranker(Protocol):
    def rerank(self, query: MedicalQuery, evidence: Sequence[Evidence]) -> List[Evidence]: ...


class HeuristicReranker:
    """Dependency-free baseline; replace with a validated cross-encoder in production."""

    def rerank(self, query: MedicalQuery, evidence: Sequence[Evidence]) -> List[Evidence]:
        terms = set(tokenize(query.normalized))
        ranked = list(evidence)
        for item in ranked:
            chunk_terms = set(tokenize(item.chunk.text))
            overlap = len(terms & chunk_terms) / max(1, len(terms))
            authority = float(item.document.metadata.get("authority_score", 0.0))
            item.rerank_score = item.rrf_score * 20 + overlap + min(1.0, max(0.0, authority)) * 0.1
        return sorted(ranked, key=lambda item: item.rerank_score or 0.0, reverse=True)


class CallableReranker:
    """Adapter for a cross-encoder function returning one score per passage."""

    def __init__(self, scorer: Callable[[str, Sequence[str]], Sequence[float]]) -> None:
        self.scorer = scorer

    def rerank(self, query: MedicalQuery, evidence: Sequence[Evidence]) -> List[Evidence]:
        ranked = list(evidence)
        scores = list(self.scorer(query.normalized, [item.chunk.contextual_text for item in ranked]))
        if len(scores) != len(ranked):
            raise ValueError("Reranker returned the wrong number of scores.")
        for item, score in zip(ranked, scores):
            item.rerank_score = float(score)
        return sorted(ranked, key=lambda item: item.rerank_score or 0.0, reverse=True)
