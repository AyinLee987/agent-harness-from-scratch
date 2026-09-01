"""Tests for query-intent classification + sub-question decomposition.

``MedicalQueryPlanner.plan()`` computes a ``subquestions`` field by
splitting on punctuation, but nothing downstream ever reads it -- this
module's ``LLMQueryDecomposer`` + ``RAGPipeline`` wiring is what makes it
live. Fully offline: a scripted stand-in LLM drives ``decompose()``
deterministically (real classification quality needs a real model --
that's what ``examples/rag_multihop_eval.py`` measures).
"""

from __future__ import annotations

import json

import pytest

from agent import LLMResponse, MockLLM, Usage
from agent.rag import (
    BM25Retriever,
    CitationCounter,
    DenseRetriever,
    EvidenceStatus,
    InMemoryRAGRepository,
    LLMQueryDecomposer,
    MedicalParentChildChunker,
    RAGConfig,
    RAGContextProvider,
    RAGIngestionService,
    RAGPipeline,
    create_rag_search_tool,
    format_evidence_context,
)
from agent.rag.decomposition import QueryDecomposition, _parse
from agent.rag.models import EvidenceBundle, MedicalQuery
from agent.rag.pipeline import _merge_parallel_bundles
from agent.memory import LLMEmbeddingProvider

PREGNANCY_TEXT = """# 妊娠用药指南
## 禁忌
妊娠患者禁忌使用示例药物 A。
"""

DOSE_TEXT = """# 成人剂量指南
## 用法用量
示例药物 A 成人推荐剂量为每日 5 mg。
"""

HYPERTENSION_TEXT = """# 高血压一线用药
## 推荐意见
成人高血压一线推荐使用示例药物 A。证据等级：A。
"""

ELDERLY_DOSE_TEXT = """# 老年人用药剂量
## 特殊人群
示例药物 A 在老年人群体中的推荐剂量为每日 2.5 mg，从低剂量起始。
"""


def _runtime():
    repository = InMemoryRAGRepository()
    embeddings = LLMEmbeddingProvider(MockLLM(), model_id="test:hash")
    bm25 = BM25Retriever(repository)
    dense = DenseRetriever(repository, embeddings)
    ingestion = RAGIngestionService(
        repository, MedicalParentChildChunker(target_tokens=60, min_tokens=15, max_tokens=100),
        [bm25, dense],
    )
    return repository, bm25, dense, ingestion


class _ScriptedDecomposerLLM(MockLLM):
    """Returns a fixed JSON decomposition verdict regardless of input."""

    def __init__(self, payload: dict) -> None:
        super().__init__()
        self._payload = payload

    def chat(self, messages, tools=None):
        return LLMResponse(content=json.dumps(self._payload, ensure_ascii=False), usage=Usage(1, 1))


class _RaisingLLM(MockLLM):
    def chat(self, messages, tools=None):
        raise RuntimeError("classification endpoint unavailable")


# ---------------------------------------------------------------------------
# _parse() -- deterministic, no LLM involved
# ---------------------------------------------------------------------------


def test_parse_accepts_a_well_formed_parallel_verdict():
    raw = json.dumps({
        "mode": "parallel",
        "subquestions": ["问题一", "问题二"],
        "reasoning": "两个独立子问题",
    }, ensure_ascii=False)
    result = _parse(raw)
    assert result.mode == "parallel"
    assert result.subquestions == ["问题一", "问题二"]


def test_parse_fails_closed_on_garbage_text():
    assert _parse("not json at all") == QueryDecomposition()


def test_parse_fails_closed_on_unknown_mode():
    raw = json.dumps({"mode": "banana", "subquestions": []})
    assert _parse(raw) == QueryDecomposition()


def test_parse_fails_closed_on_parallel_with_fewer_than_two_subquestions():
    """A 'parallel' verdict with 0-1 sub-questions isn't actionable."""
    raw = json.dumps({"mode": "parallel", "subquestions": ["only one"]})
    result = _parse(raw)
    assert result.mode == "single_hop"
    assert result.subquestions == []


def test_parse_discards_subquestions_for_non_parallel_modes():
    """Even if the model hallucinates subquestions for sequential/single_hop,
    they must not leak through -- sequential ones would be fabricated content
    for a hop that hasn't run yet."""
    raw = json.dumps({"mode": "sequential", "subquestions": ["fabricated hop 2"]})
    result = _parse(raw)
    assert result.mode == "sequential"
    assert result.subquestions == []


# ---------------------------------------------------------------------------
# LLMQueryDecomposer -- the LLM call wrapper
# ---------------------------------------------------------------------------


def test_decomposer_fails_closed_when_the_llm_call_raises():
    decomposer = LLMQueryDecomposer(_RaisingLLM())
    assert decomposer.decompose("任意问题") == QueryDecomposition()


def test_decomposer_extracts_json_even_with_surrounding_prose():
    llm = _ScriptedDecomposerLLM({"mode": "single_hop", "subquestions": [], "reasoning": "ok"})

    class ChattyLLM(MockLLM):
        def chat(self, messages, tools=None):
            return LLMResponse(
                content='这是我的判断：\n{"mode": "single_hop", "subquestions": [], "reasoning": "ok"}\n谢谢',
                usage=Usage(1, 1),
            )

    decomposer = LLMQueryDecomposer(ChattyLLM())
    result = decomposer.decompose("任意问题")
    assert result.mode == "single_hop"


# ---------------------------------------------------------------------------
# RAGPipeline wiring
# ---------------------------------------------------------------------------


def test_pipeline_without_a_decomposer_behaves_exactly_as_before():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="preg", title="妊娠指南", content=PREGNANCY_TEXT)
    pipeline = RAGPipeline(repository, bm25, dense, config=RAGConfig(minimum_evidence=1))
    bundle = pipeline.retrieve("妊娠患者能用示例药物 A 吗？")
    assert bundle.query.mode == "single_hop"
    assert bundle.decomposition_hint == ""


def test_parallel_verdict_retrieves_each_subquestion_and_merges_evidence():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="preg", title="妊娠指南", content=PREGNANCY_TEXT)
    ingestion.ingest_text(logical_id="dose", title="剂量指南", content=DOSE_TEXT)

    decomposer = LLMQueryDecomposer(_ScriptedDecomposerLLM({
        "mode": "parallel",
        "subquestions": ["妊娠患者能用示例药物 A 吗？", "示例药物 A 成人推荐剂量是多少？"],
        "reasoning": "两个独立子问题",
    }))
    pipeline = RAGPipeline(
        repository, bm25, dense, config=RAGConfig(minimum_evidence=1), decomposer=decomposer,
    )
    bundle = pipeline.retrieve("妊娠患者能用示例药物 A 吗？另外成人推荐剂量是多少？")

    assert bundle.query.mode == "parallel"
    assert len(bundle.query.subquestions) == 2
    # Evidence from *both* documents must be present -- this is the whole
    # point: a single unsplit retrieval would likely favor one document's
    # chunks over the other's for a compound query.
    document_titles = {item.document.title for item in bundle.evidence}
    assert document_titles == {"妊娠指南", "剂量指南"}
    assert bundle.status == EvidenceStatus.SUFFICIENT


def test_parallel_merge_is_only_sufficient_if_every_subquestion_is():
    """Unit-tests _merge_parallel_bundles directly with hand-built
    sub-bundles (status is all the merge logic actually inspects) --
    going through a real tiny corpus is flaky here: with only one document
    indexed, RRF's rank-1 fallback makes *any* query look "sufficient"
    (there's nothing else to rank against), which isn't what this test is
    about. examples/rag_multihop_eval.py covers the realistic end-to-end
    case against a real corpus + real LLM.
    """
    merged_query = MedicalQuery(
        original="q", normalized="q", lexical_queries=["q"], semantic_queries=["q"],
        subquestions=["妊娠患者能用示例药物 A 吗？", "示例药物 B 的化学结构式是什么？"],
        mode="parallel",
    )

    def _sub_query(text: str) -> MedicalQuery:
        return MedicalQuery(original=text, normalized=text, lexical_queries=[text], semantic_queries=[text])

    answered = EvidenceBundle(EvidenceStatus.SUFFICIENT, _sub_query("妊娠患者能用示例药物 A 吗？"), evidence=[])
    unanswered = EvidenceBundle(
        EvidenceStatus.INSUFFICIENT, _sub_query("示例药物 B 的化学结构式是什么？"),
        missing_information=["Not enough independently matching evidence was retrieved."],
    )
    merged = _merge_parallel_bundles(merged_query, [answered, unanswered])
    assert merged.status != EvidenceStatus.SUFFICIENT
    assert any("化学结构式" in item for item in merged.missing_information)


def test_sequential_verdict_runs_normal_retrieval_and_attaches_a_hint():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="htn", title="一线用药", content=HYPERTENSION_TEXT)
    ingestion.ingest_text(logical_id="elderly", title="老年剂量", content=ELDERLY_DOSE_TEXT)

    decomposer = LLMQueryDecomposer(_ScriptedDecomposerLLM({
        "mode": "sequential", "subquestions": [], "reasoning": "需要先查出药名",
    }))
    pipeline = RAGPipeline(
        repository, bm25, dense, config=RAGConfig(minimum_evidence=1), decomposer=decomposer,
    )
    bundle = pipeline.retrieve("治疗高血压的一线药物在老年人的推荐剂量是多少？")

    assert bundle.query.mode == "sequential"
    assert bundle.decomposition_hint != ""
    assert "medical_evidence_search" in bundle.decomposition_hint


def test_format_evidence_context_renders_decomposition_fields():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="htn", title="一线用药", content=HYPERTENSION_TEXT)
    decomposer = LLMQueryDecomposer(_ScriptedDecomposerLLM({
        "mode": "sequential", "subquestions": [], "reasoning": "x",
    }))
    pipeline = RAGPipeline(repository, bm25, dense, decomposer=decomposer)
    bundle = pipeline.retrieve("治疗高血压的一线药物在老年人的推荐剂量是多少？")
    text = format_evidence_context(bundle)
    assert "问题类型: sequential" in text
    assert "多跳提示:" in text


def test_decomposer_exception_during_retrieve_falls_back_to_single_hop():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="preg", title="妊娠指南", content=PREGNANCY_TEXT)
    pipeline = RAGPipeline(
        repository, bm25, dense, config=RAGConfig(minimum_evidence=1),
        decomposer=LLMQueryDecomposer(_RaisingLLM()),
    )
    bundle = pipeline.retrieve("妊娠患者能用示例药物 A 吗？")
    assert bundle.query.mode == "single_hop"
    assert bundle.status == EvidenceStatus.SUFFICIENT


# ---------------------------------------------------------------------------
# CitationCounter -- fixes [E#] colliding across multiple retrieval passes
# in one run (the mandatory RAGContextProvider injection, plus any number
# of medical_evidence_search follow-up calls independently restarting at
# [E1] otherwise -- found running examples/rag_multihop_eval.py).
# ---------------------------------------------------------------------------


def test_format_evidence_context_with_no_counter_still_restarts_at_e1():
    """Default (no shared counter) behavior is unchanged: each call is its
    own private numbering, starting at [E1] -- this is what every existing
    caller of format_evidence_context() gets for free, no code changes."""
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="preg", title="妊娠指南", content=PREGNANCY_TEXT)
    pipeline = RAGPipeline(repository, bm25, dense, config=RAGConfig(minimum_evidence=1))

    first = format_evidence_context(pipeline.retrieve("妊娠患者能用示例药物 A 吗？"))
    second = format_evidence_context(pipeline.retrieve("妊娠患者能用示例药物 A 吗？"))
    assert "[E1]" in first
    assert "[E1]" in second  # restarted, not "[E2]" -- no counter was shared


def test_citation_counter_shared_across_calls_numbers_continuously():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="preg", title="妊娠指南", content=PREGNANCY_TEXT)
    ingestion.ingest_text(logical_id="dose", title="剂量指南", content=DOSE_TEXT)
    pipeline = RAGPipeline(repository, bm25, dense, config=RAGConfig(minimum_evidence=1))
    counter = CitationCounter()

    first = format_evidence_context(pipeline.retrieve("妊娠患者能用示例药物 A 吗？"), counter)
    second = format_evidence_context(pipeline.retrieve("成人推荐剂量是多少？"), counter)

    assert "[E1]" in first
    # The second call's citations must not reuse any index the first call
    # already used -- no [E1] in the second block, it continues where the
    # first left off.
    assert "[E1]" not in second
    assert any(f"[E{n}]" in second for n in (2, 3, 4))


def test_citation_counter_take_with_zero_evidence_does_not_advance():
    counter = CitationCounter()
    assert counter.take(0) == 1
    assert counter.take(0) == 1  # still 1 -- nothing was reserved
    assert counter.take(2) == 1  # first real reservation still starts at 1
    assert counter.take(1) == 3  # advanced past the two just reserved


def test_context_provider_and_search_tool_share_citations_with_one_counter():
    """The actual bug scenario: a mandatory injection followed by a
    follow-up medical_evidence_search call, wired the way app/server.py
    wires them -- same CitationCounter passed to both."""
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="preg", title="妊娠指南", content=PREGNANCY_TEXT)
    ingestion.ingest_text(logical_id="dose", title="剂量指南", content=DOSE_TEXT)
    pipeline = RAGPipeline(repository, bm25, dense, config=RAGConfig(minimum_evidence=1))

    counter = CitationCounter()
    provider = RAGContextProvider(pipeline, citation_counter=counter)
    tool = create_rag_search_tool(pipeline, citation_counter=counter)

    [injected] = provider.prepare("妊娠患者能用示例药物 A 吗？")
    injected_text = injected["content"]
    assert "[E1]" in injected_text

    follow_up_text = tool.run(query="成人推荐剂量是多少？")
    # This is the collision the bug allowed: without a shared counter this
    # would also contain "[E1]", labeling different evidence with the same
    # tag a model can't tell apart.
    assert "[E1]" not in follow_up_text
