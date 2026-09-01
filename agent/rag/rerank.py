"""Replaceable reranking layer."""

from __future__ import annotations

import json
from typing import Callable, List, Protocol, Sequence

from ..llm import BaseLLM
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


_LLM_RERANK_PROMPT = """You are scoring how relevant each candidate passage is to a search query, for a retrieval system. Score every passage independently on a 0-10 scale (10 = directly and fully answers the query, 0 = completely irrelevant to it).

Respond with ONLY a JSON array of numbers, one real score per passage, in the same order as the passages -- no explanation, no markdown fences, no placeholder text describing the array, nothing else but the array of actual numbers itself.

### Example

Query: What foods lower cholesterol?

Candidate passages:
[1] Oatmeal contains soluble fiber that has been shown to reduce LDL cholesterol levels when eaten regularly.
[2] The Eiffel Tower was completed in 1889 and is located in Paris, France.
[3] Regular exercise improves cardiovascular health but this passage does not mention cholesterol specifically.
[4] Walnuts and almonds contain unsaturated fats that can help lower bad cholesterol.
[5] A study of urban transportation patterns in the 1990s found commuting times increased.

Your response:
[9, 0, 3, 8, 0]

### Now score this one

Query: {query}

Candidate passages:
{passages}

Your response (a JSON array of exactly {n} real numbers, nothing else):"""


def _parse_llm_rerank_scores(text: str, expected_n: int) -> List[float]:
    """Extract a JSON array of exactly ``expected_n`` numbers from ``text``.

    Tolerant of the model wrapping the array in a markdown code fence or
    adding a stray sentence before/after it -- takes the substring between
    the first ``[`` and the last ``]`` rather than requiring the whole
    response to be valid JSON on its own.
    """

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON array found in LLM rerank response: {text[:200]!r}")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, list) or len(parsed) != expected_n:
        raise ValueError(f"Expected {expected_n} scores, got: {parsed!r}")
    return [float(x) for x in parsed]


class LLMReranker:
    """Asks a chat LLM to score each candidate's relevance to the query,
    instead of ``HeuristicReranker``'s lexical-overlap heuristic.

    Measured on BEIR NFCorpus (323 real published queries -- see the
    companion evaluation project's
    ``benchmarks/rag_recall_beir/RESULTS_llm_rerank.md``): beats plain RRF
    fusion and ``HeuristicReranker`` on *every* metric, clearing paired
    bootstrap significance (p<0.05, mostly p<0.001) on Recall@5/10 and
    nDCG@5/10 against both -- not significantly on MRR, the noisiest
    metric on that benchmark throughout. Real cost: ~2.2s/query (one LLM
    call per retrieval), 6-12x slower than the other rerankers, since
    that's real generation latency, not a local computation. Opt-in for a
    reason (``ENABLE_LLM_RERANKER`` in ``.env.example``), not a default.

    The prompt is deliberately one-shot (a full worked example), not
    zero-shot: an earlier zero-shot version (instructions + a one-line
    format description, no worked example) had a measured 41%
    degenerate-response rate scoring ~30 candidates in one call on
    ``deepseek-chat`` -- the model would echo the prompt's own format
    description back verbatim (literally the string
    ``"[json array of 30 numbers]"``) instead of doing the task. Swapping
    in a full worked example brought that to 0/323 on the same benchmark.

    Raises on an unparseable response rather than fabricating scores or
    silently degrading -- ``RAGPipeline._retrieve_single`` already catches
    any reranker exception and falls back to plain RRF order for that one
    query (see ``degraded_components``), which is the right place for that
    fallback to live, not duplicated here.
    """

    def __init__(self, llm: BaseLLM, *, max_passage_chars: int = 300) -> None:
        self.llm = llm
        self.max_passage_chars = max_passage_chars
        self._delegate = CallableReranker(self._score)

    def rerank(self, query: MedicalQuery, evidence: Sequence[Evidence]) -> List[Evidence]:
        return self._delegate.rerank(query, evidence)

    def _score(self, query_text: str, passages: Sequence[str]) -> List[float]:
        truncated = [p[: self.max_passage_chars] for p in passages]
        numbered = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(truncated))
        prompt = _LLM_RERANK_PROMPT.format(query=query_text, passages=numbered, n=len(truncated))
        response = self.llm.chat([{"role": "user", "content": prompt}])
        return _parse_llm_rerank_scores(response.content or "", len(truncated))
