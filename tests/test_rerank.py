"""Tests for the replaceable reranking layer, focused on ``LLMReranker``.

Fully offline: a scripted MockLLM stands in for the reranking model, same
pattern as tests/test_rag_decomposition.py's LLMQueryDecomposer tests.
Real reranking quality against a live model is what the companion
evaluation project's benchmarks/rag_recall_beir/RESULTS_llm_rerank.md
measures.
"""

from __future__ import annotations

import pytest

from agent import LLMReranker, LLMResponse, MockLLM, Usage
from agent.rag import (
    BM25Retriever,
    DenseRetriever,
    InMemoryRAGRepository,
    MedicalParentChildChunker,
    RAGConfig,
    RAGIngestionService,
    RAGPipeline,
)
from agent.rag.rerank import _parse_llm_rerank_scores
from agent.memory import LLMEmbeddingProvider

DOC_A = """# Oatmeal and Cholesterol
## Fiber
Oatmeal contains soluble fiber that lowers LDL cholesterol when eaten regularly.
"""

DOC_B = """# Paris Landmarks
## History
The Eiffel Tower was completed in 1889 and is a famous Paris landmark.
"""


def _runtime():
    repository = InMemoryRAGRepository()
    embeddings = LLMEmbeddingProvider(MockLLM(), model_id="test:hash")
    bm25 = BM25Retriever(repository)
    dense = DenseRetriever(repository, embeddings)
    ingestion = RAGIngestionService(
        repository,
        MedicalParentChildChunker(target_tokens=80, min_tokens=20, max_tokens=120),
        [bm25, dense],
    )
    ingestion.ingest_text(logical_id="oatmeal", title="Oatmeal and Cholesterol", content=DOC_A)
    ingestion.ingest_text(logical_id="paris", title="Paris Landmarks", content=DOC_B)
    return repository, bm25, dense


class _ScoringLLM(MockLLM):
    """Returns a fixed response string regardless of the prompt -- the
    reranker only cares that *a* response with the right shape comes back,
    not that this stand-in actually reads the prompt."""

    def __init__(self, content: str) -> None:
        super().__init__()
        self.content = content
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        return LLMResponse(content=self.content, usage=Usage(1, 1))


# ---------------------------------------------------------------------------
# _parse_llm_rerank_scores
# ---------------------------------------------------------------------------
def test_parse_scores_accepts_a_plain_json_array():
    assert _parse_llm_rerank_scores("[7, 2, 9]", 3) == [7.0, 2.0, 9.0]


def test_parse_scores_tolerates_a_markdown_fence_and_stray_text():
    assert _parse_llm_rerank_scores("```json\n[7, 2, 9]\n```", 3) == [7.0, 2.0, 9.0]
    assert _parse_llm_rerank_scores("Here you go: [7, 2, 9]. Done.", 3) == [7.0, 2.0, 9.0]


def test_parse_scores_rejects_wrong_length():
    with pytest.raises(ValueError, match="Expected 3 scores"):
        _parse_llm_rerank_scores("[7, 2]", 3)


def test_parse_scores_rejects_missing_array():
    with pytest.raises(ValueError, match="No JSON array found"):
        _parse_llm_rerank_scores("I decline to score these.", 3)


def test_parse_scores_rejects_the_degenerate_placeholder_response():
    """The exact failure mode that motivated the one-shot prompt rewrite
    (see LLMReranker's docstring): the model echoing the prompt's own
    format description back instead of doing the task."""
    with pytest.raises(ValueError):
        _parse_llm_rerank_scores("[json array of 3 numbers]", 3)


# ---------------------------------------------------------------------------
# LLMReranker, wired into a real RAGPipeline
# ---------------------------------------------------------------------------
def test_llm_reranker_reorders_evidence_by_the_scores_it_returns():
    repository, bm25, dense = _runtime()
    # Score the second-ranked candidate highest, regardless of RRF order.
    llm = _ScoringLLM("[1, 9]")
    pipeline = RAGPipeline(
        repository, bm25, dense, reranker=LLMReranker(llm),
        config=RAGConfig(minimum_evidence=1),
    )

    bundle = pipeline.retrieve("Eiffel Tower Paris")

    assert llm.calls == 1
    assert len(bundle.evidence) == 2
    # Whichever candidate the scorer ranked highest (score 9) must be first.
    assert bundle.evidence[0].rerank_score == 9.0
    assert bundle.evidence[1].rerank_score == 1.0
    assert not bundle.degraded_components


def test_llm_reranker_failure_degrades_to_plain_rrf_order_not_a_crash():
    repository, bm25, dense = _runtime()
    llm = _ScoringLLM("[json array of 2 numbers]")  # the degenerate response
    pipeline = RAGPipeline(
        repository, bm25, dense, reranker=LLMReranker(llm),
        config=RAGConfig(minimum_evidence=1),
    )

    bundle = pipeline.retrieve("Eiffel Tower Paris")

    assert bundle.evidence  # retrieval still returns results
    assert "reranker" in bundle.degraded_components
    # Fell back to un-reranked (RRF) order -- no rerank_score was ever set.
    assert all(item.rerank_score is None for item in bundle.evidence)


def test_llm_reranker_truncates_long_passages_before_prompting():
    repository, bm25, dense = _runtime()
    captured = {}

    class _CapturingLLM(MockLLM):
        def chat(self, messages, tools=None):
            captured["prompt"] = messages[0]["content"]
            return LLMResponse(content="[5, 5]", usage=Usage(1, 1))

    pipeline = RAGPipeline(
        repository, bm25, dense,
        reranker=LLMReranker(_CapturingLLM(), max_passage_chars=10),
        config=RAGConfig(minimum_evidence=1),
    )
    pipeline.retrieve("Eiffel Tower Paris")

    # Every candidate line in the prompt should reflect the 10-char cap,
    # not the full chunk text.
    body = captured["prompt"].split("### Now score this one")[1]
    for line in body.splitlines():
        if line.startswith("["):
            passage_text = line.split("] ", 1)[1]
            assert len(passage_text) <= 10
