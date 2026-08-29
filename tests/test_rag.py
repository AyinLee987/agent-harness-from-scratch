from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

from agent import MockLLM, ReActAgent, ToolRegistry
from agent.memory import LLMEmbeddingProvider
from agent.rag import (
    BM25Retriever, DenseRetriever, DocumentStatus, EvidenceStatus,
    InMemoryRAGRepository, MedicalParentChildChunker, RAGConfig,
    RAGContextProvider, RAGIngestionService, RAGPipeline,
    RetrievalFilters, SQLiteRAGRepository, create_rag_search_tool,
)


TEXT = """# 高血压治疗
## 成人推荐意见
推荐意见：成人高血压患者应结合风险分层选择治疗。证据等级：A。
用法用量：示例药物每日 5 mg；实际用药必须遵医嘱。
## 特殊人群
妊娠患者禁忌使用该示例药物。
| 人群 | 建议 |
| --- | --- |
| 老年 | 从低剂量开始 |
"""


def _runtime(repository=None):
    repository = repository or InMemoryRAGRepository()
    embeddings = LLMEmbeddingProvider(MockLLM(), model_id="test:hash")
    bm25 = BM25Retriever(repository)
    dense = DenseRetriever(repository, embeddings)
    ingestion = RAGIngestionService(
        repository, MedicalParentChildChunker(target_tokens=80, min_tokens=20, max_tokens=120),
        [bm25, dense],
    )
    return repository, bm25, dense, ingestion


def test_medical_chunking_preserves_structure_and_parent_links():
    repository, _, _, ingestion = _runtime()
    result = ingestion.ingest_text(
        logical_id="guide/hypertension", title="指南", content=TEXT,
        publisher="权威学会", document_type="guideline", jurisdiction="CN",
    )
    children = [item for item in result.chunks if item.chunk_type == "child"]
    parents = {item.id for item in result.chunks if item.chunk_type == "parent"}
    assert children
    assert all(item.parent_chunk_id in parents for item in children)
    assert any("推荐意见" in item.text and item.evidence_grade == "A" for item in children)
    assert any("表头:" in item.text and "表行:" in item.text for item in children)
    assert repository.get_document(result.document.id).status == DocumentStatus.ACTIVE


def test_ingestion_deduplicates_and_atomically_supersedes_versions():
    repository, _, _, ingestion = _runtime()
    first = ingestion.ingest_text(logical_id="guide/x", title="X", content=TEXT, version="1")
    duplicate = ingestion.ingest_text(logical_id="guide/x", title="X", content=TEXT, version="1")
    second = ingestion.ingest_text(
        logical_id="guide/x", title="X", content=TEXT + "\n新增复审内容。", version="2"
    )
    assert duplicate.skipped
    assert repository.get_document(first.document.id).status == DocumentStatus.SUPERSEDED
    assert second.document.status == DocumentStatus.ACTIVE
    assert second.document.supersedes_id == first.document.id
    assert all(item.document_id == second.document.id for item in repository.active_chunks())


def test_hybrid_retrieval_filters_and_citations():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(
        logical_id="guide/hypertension", title="高血压指南", content=TEXT,
        publisher="权威学会", document_type="guideline", jurisdiction="CN",
    )
    pipeline = RAGPipeline(repository, bm25, dense, config=RAGConfig(minimum_evidence=1))
    bundle = pipeline.retrieve("妊娠患者有什么禁忌？")
    assert bundle.status == EvidenceStatus.SUFFICIENT
    assert bundle.evidence
    assert bundle.evidence[0].citation.document_id
    assert bundle.evidence[0].citation.chunk_id
    blocked = pipeline.retrieve("妊娠患者有什么禁忌？", RetrievalFilters(jurisdiction="US"))
    assert blocked.status == EvidenceStatus.INSUFFICIENT


def test_one_retriever_can_fail_without_losing_available_evidence():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="g", title="G", content=TEXT)

    class BrokenDense:
        name = "dense"
        def search(self, query, limit=20):
            raise RuntimeError("embedding endpoint unavailable")

    bundle = RAGPipeline(repository, bm25, BrokenDense()).retrieve("成人高血压推荐意见")
    assert bundle.status == EvidenceStatus.SUFFICIENT
    assert bundle.degraded_components == ["dense"]


def test_evidence_gate_reports_stale_and_conflicting_sources():
    repository, bm25, dense, ingestion = _runtime()
    old = datetime.now(timezone.utc) - timedelta(days=800)
    ingestion.ingest_text(
        logical_id="old", title="旧指南", content="# 建议\n目标血压建议低于 140 mmHg。",
        published_at=old, metadata={"conflict_group": "bp-target", "stance": "140"},
    )
    ingestion.ingest_text(
        logical_id="new", title="另一指南", content="# 建议\n目标血压建议低于 130 mmHg。",
        published_at=old, metadata={"conflict_group": "bp-target", "stance": "130"},
    )
    conflict = RAGPipeline(repository, bm25, dense).retrieve("目标血压建议低于多少？")
    assert conflict.status == EvidenceStatus.CONFLICTING
    assert conflict.conflicts[0].conflict_group == "bp-target"

    stale = RAGPipeline(
        repository, bm25, dense, config=RAGConfig(max_evidence_age_days=365)
    ).retrieve("目标血压")
    # Conflict has higher priority; after selecting one source, freshness is explicit.
    one_source = RAGPipeline(
        repository, bm25, dense,
        config=RAGConfig(max_evidence_age_days=365, evidence_limit=1),
    ).retrieve("目标血压")
    assert stale.status == EvidenceStatus.CONFLICTING
    assert one_source.status == EvidenceStatus.STALE


def test_rag_context_is_injected_before_first_model_call():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="g", title="G", content=TEXT)
    pipeline = RAGPipeline(repository, bm25, dense)

    class CapturingLLM(MockLLM):
        def chat(self, messages, tools=None):
            self.messages = messages
            return super().chat(messages, tools)

    llm = CapturingLLM()
    agent = ReActAgent(
        llm=llm, tools=ToolRegistry(), context_providers=[RAGContextProvider(pipeline)]
    )
    agent.run("妊娠患者有什么禁忌？")
    assert llm.messages[0]["role"] == "system"
    assert llm.messages[1]["role"] == "system"
    assert "证据状态:" in llm.messages[1]["content"]
    assert "chunk_id=" in llm.messages[1]["content"]
    assert llm.messages[2]["role"] == "user"


def test_follow_up_search_is_a_recoverable_agent_tool():
    repository, bm25, dense, ingestion = _runtime()
    ingestion.ingest_text(logical_id="g", title="G", content=TEXT)
    tool = create_rag_search_tool(RAGPipeline(repository, bm25, dense))
    output = tool.run(query="成人用量")
    assert tool.name == "medical_evidence_search"
    assert "证据状态:" in output
    assert "[E1]" in output


def test_sqlite_repository_round_trip_and_publish(tmp_path: Path):
    repository = SQLiteRAGRepository(tmp_path / "rag.sqlite")
    _, bm25, dense, ingestion = _runtime(repository)
    result = ingestion.ingest_text(logical_id="g", title="G", content=TEXT)
    repository.close()

    reopened = SQLiteRAGRepository(tmp_path / "rag.sqlite")
    try:
        loaded = reopened.get_document(result.document.id)
        assert loaded is not None and loaded.status == DocumentStatus.ACTIVE
        assert reopened.active_chunks()
    finally:
        reopened.close()


def test_ingestion_failure_never_publishes_document():
    repository = InMemoryRAGRepository()

    class BrokenIndexer:
        def index_document(self, document, chunks):
            raise RuntimeError("index failed")

    ingestion = RAGIngestionService(repository, MedicalParentChildChunker(), [BrokenIndexer()])
    with pytest.raises(RuntimeError, match="index failed"):
        ingestion.ingest_text(logical_id="g", title="G", content=TEXT)
    assert repository.list_documents()[0].status == DocumentStatus.FAILED
    assert repository.active_chunks() == []
