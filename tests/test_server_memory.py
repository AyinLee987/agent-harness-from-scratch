"""Server-level checks for the ENABLE_LONG_TERM_MEMORY wiring.

Fully offline (MockLLM for chat, MockLLM's deterministic hash embedding for
the vector index) -- no API keys needed, matching the rest of the RAG/
memory-manager test suite's philosophy. ``_start_memory()`` itself (env
vars -> a real OpenAICompatibleEmbeddingProvider) is exercised manually
against a live provider; here MEMORY_MANAGER is installed directly so the
wiring around it -- tool registration, memory_manager/namespace/subject_id
threading into ReActAgent, and end-to-end persistence through /api/run --
is what's under test.
"""

from __future__ import annotations

import asyncio

from app import server
from agent import (
    DefaultMemoryPolicy,
    ExplicitRequestMemoryExtractor,
    LLMResponse,
    MemoryManager,
    MemoryStatus,
    MockLLM,
    Usage,
)
from agent.memory import LLMEmbeddingProvider


def _install_memory_manager() -> MemoryManager:
    manager = MemoryManager(
        LLMEmbeddingProvider(MockLLM(), model_id="test:hash"),
        extractor=ExplicitRequestMemoryExtractor(),
        policy=DefaultMemoryPolicy(),
    )
    server.MEMORY_MANAGER = manager
    return manager


def test_memory_search_tool_is_absent_when_memory_is_disabled():
    assert server.MEMORY_MANAGER is None  # nothing in this test enabled it
    orchestrator, leader = server._build_leader_runtime()
    try:
        assert "memory_search" not in leader.tools.names()
        assert leader.memory_manager is None
    finally:
        orchestrator.close()


def test_memory_search_tool_and_hook_are_wired_when_manager_is_set():
    _install_memory_manager()
    try:
        orchestrator, leader = server._build_leader_runtime()
        try:
            assert "memory_search" in leader.tools.names()
            assert leader.memory_manager is server.MEMORY_MANAGER
            assert leader.memory_namespace == server.MEMORY_NAMESPACE
            assert leader.memory_subject_id == server.MEMORY_SUBJECT_ID
        finally:
            orchestrator.close()
    finally:
        server.MEMORY_MANAGER = None


def test_explicit_remember_requests_persist_and_conflicting_facts_coexist(monkeypatch):
    """Two separate /api/run calls (two 'conversations', no conversation_id
    shared between them) -- 'remember I like coffee' then 'remember I don't
    like coffee'. Both must land as separate ACTIVE records: there is no
    automatic conflict resolution by design (agent/memory/policy.py never
    supersedes on a plain 'remember ...' -- only MemoryManager.supersede()
    does, and nothing calls it here), so a contradiction is stored, not
    silently merged or overwritten.
    """
    manager = _install_memory_manager()
    try:
        class NotingLLM(MockLLM):
            def chat(self, messages, tools=None):
                return LLMResponse(content="Noted.", usage=Usage(1, 1))

        monkeypatch.setattr(server, "_build_llm", NotingLLM)

        asyncio.run(server.run(server.RunRequest(task="请记住我喜欢咖啡")))
        asyncio.run(server.run(server.RunRequest(task="请记住我不喜欢咖啡")))

        records = manager.repository.list_records(
            namespace=server.MEMORY_NAMESPACE,
            subject_id=server.MEMORY_SUBJECT_ID,
            status=MemoryStatus.ACTIVE,
        )
        assert {r.content for r in records} == {"我喜欢咖啡", "我不喜欢咖啡"}
        assert len(records) == 2  # not deduplicated, not superseded
    finally:
        server.MEMORY_MANAGER = None


def test_ordinary_conversation_does_not_write_memory(monkeypatch):
    """The extractor only fires on an explicit '记住/remember' message --
    an ordinary statement of preference must not silently get persisted."""
    manager = _install_memory_manager()
    try:
        class ChattyLLM(MockLLM):
            def chat(self, messages, tools=None):
                return LLMResponse(content="Got it.", usage=Usage(1, 1))

        monkeypatch.setattr(server, "_build_llm", ChattyLLM)
        asyncio.run(server.run(server.RunRequest(task="我喜欢咖啡")))

        records = manager.repository.list_records(
            namespace=server.MEMORY_NAMESPACE, subject_id=server.MEMORY_SUBJECT_ID,
        )
        assert records == []
    finally:
        server.MEMORY_MANAGER = None


def test_api_memory_endpoint_404s_when_disabled():
    import pytest
    from fastapi import HTTPException

    assert server.MEMORY_MANAGER is None
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.list_memory())
    assert exc_info.value.status_code == 404


def test_api_memory_endpoint_lists_active_records(monkeypatch):
    manager = _install_memory_manager()
    try:
        class NotingLLM(MockLLM):
            def chat(self, messages, tools=None):
                return LLMResponse(content="Noted.", usage=Usage(1, 1))

        monkeypatch.setattr(server, "_build_llm", NotingLLM)
        asyncio.run(server.run(server.RunRequest(task="请记住我喜欢咖啡")))

        body = asyncio.run(server.list_memory())
        assert body["namespace"] == server.MEMORY_NAMESPACE
        assert body["subject_id"] == server.MEMORY_SUBJECT_ID
        assert [r["content"] for r in body["records"]] == ["我喜欢咖啡"]
    finally:
        server.MEMORY_MANAGER = None
