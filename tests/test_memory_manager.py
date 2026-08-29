from __future__ import annotations

from datetime import timedelta

import pytest

from agent import (
    ExplicitRequestMemoryExtractor,
    InMemorySessionStore,
    LongTermMemory,
    MemoryCandidate,
    MemoryKind,
    MemoryManager,
    MemoryProtectedError,
    MemoryStatus,
    MockLLM,
    OpenAICompatibleEmbeddingProvider,
    ReActAgent,
    RetentionPolicy,
    RunCompletedEvent,
    SQLiteMemoryRepository,
    Sensitivity,
    SummarySnapshot,
    ToolRegistry,
)
from agent.memory.models import utc_now


class TinyEmbeddingProvider:
    model_id = "test:tiny-v1"
    dimension = 4

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [
            float(lowered.count("alpha") + lowered.count("阿尔法")),
            float(lowered.count("beta") + lowered.count("贝塔")),
            float(lowered.count("chinese") + lowered.count("中文")),
            1.0,
        ]


def candidate(content: str, **kwargs) -> MemoryCandidate:
    return MemoryCandidate(
        content=content,
        kind=kwargs.pop("kind", MemoryKind.USER_FACT),
        source_type=kwargs.pop("source_type", "user_message"),
        explicit_user_request=kwargs.pop("explicit_user_request", True),
        **kwargs,
    )


def event(task: str, *, success: bool = True) -> RunCompletedEvent:
    return RunCompletedEvent(
        run_id="run-1",
        task=task,
        answer="The model answer must not become the memory source.",
        success=success,
        stop_reason="finished" if success else "fatal_tool_error",
    )


def test_default_manager_does_not_persist_completed_runs():
    manager = MemoryManager(TinyEmbeddingProvider())
    assert manager.on_run_completed(event("ordinary question")) == []
    assert manager.repository.list_records() == []


def test_explicit_extractor_persists_user_statement_not_model_answer():
    manager = MemoryManager(
        TinyEmbeddingProvider(), extractor=ExplicitRequestMemoryExtractor()
    )
    records = manager.on_run_completed(event("Please remember that I prefer Chinese"))
    assert len(records) == 1
    assert records[0].content == "I prefer Chinese"
    assert "model answer" not in records[0].content.casefold()
    assert records[0].source_type == "user_message"


def test_failed_run_is_never_extracted():
    manager = MemoryManager(
        TinyEmbeddingProvider(), extractor=ExplicitRequestMemoryExtractor()
    )
    assert manager.on_run_completed(event("Remember alpha", success=False)) == []
    assert manager.repository.list_records() == []


def test_sensitive_memory_requires_confirmation():
    manager = MemoryManager(TinyEmbeddingProvider())
    item = candidate(
        "User reports a medicine allergy",
        sensitivity=Sensitivity.HEALTH,
    )
    assert manager.store_candidate(item) is None
    assert [pending.id for pending in manager.pending_candidates()] == [item.id]

    record = manager.confirm(item.id)
    assert record.status == MemoryStatus.ACTIVE
    assert record.metadata["confirmed"] is True


def test_model_output_is_skipped_even_when_high_importance():
    manager = MemoryManager(TinyEmbeddingProvider())
    item = candidate(
        "An unsupported model conclusion",
        source_type="model_output",
        importance=1.0,
    )
    assert manager.store_candidate(item) is None
    assert manager.repository.list_records() == []


def test_ttl_expires_but_pinned_memory_does_not():
    manager = MemoryManager(TinyEmbeddingProvider())
    expired_at = utc_now() - timedelta(minutes=1)
    temporary = candidate(
        "temporary alpha task",
        retention_policy=RetentionPolicy.TTL,
        expires_at=expired_at,
    )
    pinned = candidate(
        "important beta preference",
        retention_policy=RetentionPolicy.PINNED,
        expires_at=expired_at,
        pinned=True,
    )
    temporary_record = manager.store_candidate(temporary)
    pinned_record = manager.store_candidate(pinned)
    assert temporary_record is not None and pinned_record is not None

    assert manager.expire_due() == [temporary_record.id]
    assert manager.repository.get(temporary_record.id).status == MemoryStatus.EXPIRED
    assert manager.repository.get(pinned_record.id).status == MemoryStatus.ACTIVE


def test_superseded_memory_is_retained_but_not_recalled():
    manager = MemoryManager(TinyEmbeddingProvider())
    old = manager.store_candidate(candidate("alpha old value"))
    assert old is not None
    new = manager.supersede(old.id, candidate("beta current value"))

    assert manager.repository.get(old.id).status == MemoryStatus.SUPERSEDED
    assert manager.repository.get(new.id).status == MemoryStatus.ACTIVE
    assert new.supersedes_id == old.id
    results = manager.recall("alpha", limit=10)
    assert all(result.record.id != old.id for result in results)


def test_recall_and_tool_are_isolated_by_namespace_and_subject():
    manager = MemoryManager(TinyEmbeddingProvider())
    private = candidate(
        "alpha private fact", namespace="users", subject_id="user-a"
    )
    manager.store_candidate(private)

    assert manager.recall("alpha", namespace="users", subject_id="user-b") == []
    tool = manager.as_search_tool(namespace="users", subject_id="user-a")
    assert "alpha private fact" in tool.run(query="alpha")


def test_pinned_deletion_requires_explicit_authorization_then_can_be_purged():
    manager = MemoryManager(TinyEmbeddingProvider())
    record = manager.store_candidate(
        candidate("important alpha", retention_policy=RetentionPolicy.PINNED)
    )
    assert record is not None
    with pytest.raises(MemoryProtectedError):
        manager.tombstone(record.id, reason="automatic cleanup")

    tombstone = manager.tombstone(
        record.id,
        reason="authorized user deletion",
        allow_protected=True,
    )
    assert tombstone.status == MemoryStatus.TOMBSTONED
    assert manager.recall("alpha") == []
    purged = manager.purge_tombstones(utc_now() + timedelta(seconds=1))
    assert purged == [record.id]
    assert manager.repository.get(record.id) is None


def test_sqlite_repository_persists_structured_lifecycle(tmp_path):
    path = tmp_path / "memory.db"
    repository = SQLiteMemoryRepository(str(path))
    manager = MemoryManager(TinyEmbeddingProvider(), repository=repository)
    record = manager.store_candidate(candidate("alpha persistent fact"))
    assert record is not None
    repository.close()

    reopened = SQLiteMemoryRepository(str(path))
    loaded = reopened.get(record.id)
    assert loaded is not None
    assert loaded.content == "alpha persistent fact"
    assert loaded.embedding_model == "test:tiny-v1"
    assert loaded.status == MemoryStatus.ACTIVE
    rebuilt = MemoryManager(TinyEmbeddingProvider(), repository=reopened)
    assert rebuilt.rebuild_index() == 1
    assert rebuilt.recall("alpha")[0].record.id == record.id
    rebuilt.quarantine(record.id, "test review")
    reopened.close()

    final = SQLiteMemoryRepository(str(path))
    assert final.get(record.id).status == MemoryStatus.QUARANTINED
    final.close()


def test_chat_model_and_embedding_provider_are_decoupled():
    class ChatOnlyLLM(MockLLM):
        def embed(self, text):
            raise AssertionError("chat model embedding must not be used")

    memory = LongTermMemory(
        ChatOnlyLLM(), embedding_provider=TinyEmbeddingProvider()
    )
    memory.add("alpha fact")
    assert memory.search("alpha", k=1)[0][0] == "alpha fact"


def test_openai_compatible_embedding_provider_is_independent_and_batched():
    class Item:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

    class Embeddings:
        def create(self, *, model, input):
            assert model == "medical-embedding"
            return type(
                "Response",
                (),
                {"data": [Item(i, [float(len(text)), 1.0]) for i, text in enumerate(input)]},
            )()

    fake_client = type("Client", (), {"embeddings": Embeddings()})()
    provider = OpenAICompatibleEmbeddingProvider(
        model="medical-embedding",
        provider_name="bailian",
        client=fake_client,
    )
    assert provider.embed_documents(["a", "abcd"]) == [[1.0, 1.0], [4.0, 1.0]]
    assert provider.model_id == "bailian:medical-embedding"
    assert provider.dimension == 2


def test_legacy_long_term_memory_is_no_longer_auto_written_by_agent():
    memory = LongTermMemory(MockLLM())
    agent = ReActAgent(MockLLM(), ToolRegistry(), long_term=memory)
    result = agent.run("ordinary task")
    assert result.success
    assert len(memory) == 0


def test_memory_write_failure_does_not_fail_completed_run():
    class BrokenManager:
        def on_run_completed(self, event):
            raise RuntimeError("memory backend unavailable")

    agent = ReActAgent(
        MockLLM(), ToolRegistry(), memory_manager=BrokenManager()  # type: ignore[arg-type]
    )
    result = agent.run("ordinary task")
    assert result.success


def test_in_memory_session_store_returns_copies_and_tracks_summary():
    store = InMemorySessionStore()
    message_id = store.append_message("conversation-1", {"role": "user", "content": "hi"})
    loaded = store.load_messages("conversation-1")
    loaded[0]["content"] = "changed"
    assert store.load_messages("conversation-1")[0]["content"] == "hi"

    snapshot = SummarySnapshot("conversation-1", "summary", message_id)
    store.save_summary(snapshot)
    assert store.load_summary("conversation-1").through_message_id == message_id
