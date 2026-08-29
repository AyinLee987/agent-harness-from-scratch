from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

import server
from agent.rag import InMemoryRAGRepository, MedicalParentChildChunker, RAGIngestionService


class _Request:
    def __init__(self, token: str = "") -> None:
        self.headers = {"X-RAG-Admin-Token": token}


def _payload(content: str = "# 推荐意见\n成人患者应接受个体化评估。"):
    return server.RAGDocumentRequest(
        logical_id="guideline/demo",
        title="示例指南",
        content=content,
        publisher="测试学会",
        document_type="guideline",
        jurisdiction="CN",
        version="2026",
    )


def test_corpus_endpoint_requires_admin_token(monkeypatch):
    monkeypatch.setattr(
        server, "RAG_INGESTION",
        RAGIngestionService(InMemoryRAGRepository(), MedicalParentChildChunker()),
    )
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "secret")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(server.ingest_rag_document(_payload(), _Request("wrong")))
    assert caught.value.status_code == 401


def test_corpus_endpoint_publishes_and_reports_duplicate(monkeypatch):
    repository = InMemoryRAGRepository()
    monkeypatch.setattr(
        server, "RAG_INGESTION",
        RAGIngestionService(repository, MedicalParentChildChunker()),
    )
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "secret")
    first = asyncio.run(server.ingest_rag_document(_payload(), _Request("secret")))
    second = asyncio.run(server.ingest_rag_document(_payload(), _Request("secret")))
    assert first.status == "active"
    assert first.chunk_count > 0
    assert not first.skipped
    assert second.skipped
    assert second.document_id == first.document_id


def test_corpus_endpoint_enforces_size_limit(monkeypatch):
    monkeypatch.setattr(
        server, "RAG_INGESTION",
        RAGIngestionService(InMemoryRAGRepository(), MedicalParentChildChunker()),
    )
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "secret")
    monkeypatch.setenv("RAG_MAX_DOCUMENT_BYTES", "8")
    with pytest.raises(HTTPException) as caught:
        asyncio.run(server.ingest_rag_document(_payload("123456789"), _Request("secret")))
    assert caught.value.status_code == 413
