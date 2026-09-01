"""Contract tests for BaseVectorStore -- run against every backend.

The goal is behavioral parity: a test written once against the abstract
``BaseVectorStore`` interface and parametrized over each concrete backend
(``NumPyVectorStore``, ``SQLiteVectorStore``, ``ChromaVectorStore``, and any
future Qdrant/pgvector implementation) instead of duplicating the same
assertions per backend. Add a new backend to the ``store`` fixture's
params and it's covered by every test below for free.

The ``chroma`` backend is skipped when the optional ``chromadb`` package
isn't installed (same pattern the live-LLM tests use for missing API keys)
-- everything else here has zero optional dependencies.
"""

from __future__ import annotations

import pytest

from agent.state.store import BaseVectorStore, NumPyVectorStore, SQLiteVectorStore

try:
    import chromadb  # noqa: F401
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


@pytest.fixture(params=[
    "numpy",
    "sqlite",
    pytest.param("chroma", marks=pytest.mark.skipif(not HAS_CHROMADB, reason="chromadb not installed")),
])
def store(request, tmp_path) -> BaseVectorStore:
    if request.param == "numpy":
        return NumPyVectorStore()
    if request.param == "sqlite":
        return SQLiteVectorStore(tmp_path / "vectors.db")
    from agent.state.chroma_store import ChromaVectorStore
    return ChromaVectorStore(persist_directory=str(tmp_path / "chroma"))


def test_add_returns_a_usable_id_and_grows_length(store: BaseVectorStore):
    assert len(store) == 0
    record_id = store.add("hello", [1.0, 0.0, 0.0])
    assert record_id
    assert len(store) == 1


def test_add_with_explicit_record_id_uses_it_verbatim(store: BaseVectorStore):
    returned = store.add("hello", [1.0, 0.0, 0.0], record_id="chunk-42")
    assert returned == "chunk-42"
    assert len(store) == 1
    ids = {item["id"] for item in store.all()}
    assert ids == {"chunk-42"}


def test_add_with_a_repeated_record_id_upserts_instead_of_duplicating(store: BaseVectorStore):
    store.add("version one", [1.0, 0.0, 0.0], record_id="chunk-1")
    store.add("version two", [0.0, 1.0, 0.0], record_id="chunk-1")
    assert len(store) == 1
    hits = store.search([0.0, 1.0, 0.0], k=1)
    assert hits[0][1] == "version two"


def test_search_ranks_by_cosine_similarity_descending(store: BaseVectorStore):
    store.add("same direction", [1.0, 0.0, 0.0], record_id="a")
    store.add("orthogonal", [0.0, 1.0, 0.0], record_id="b")
    store.add("opposite", [-1.0, 0.0, 0.0], record_id="c")
    hits = store.search([1.0, 0.0, 0.0], k=3)
    assert [record_id for record_id, _text, _score in hits] == ["a", "b", "c"]
    assert hits[0][2] == pytest.approx(1.0)
    assert hits[1][2] == pytest.approx(0.0)
    assert hits[2][2] == pytest.approx(-1.0)


def test_search_respects_k(store: BaseVectorStore):
    for i in range(5):
        store.add(f"item {i}", [float(i), 1.0, 0.0], record_id=f"id-{i}")
    assert len(store.search([0.0, 1.0, 0.0], k=2)) == 2


def test_search_on_empty_store_returns_nothing(store: BaseVectorStore):
    assert store.search([1.0, 0.0, 0.0], k=5) == []


def test_delete_removes_a_record_and_reports_whether_it_existed(store: BaseVectorStore):
    record_id = store.add("hello", [1.0, 0.0, 0.0])
    assert store.delete(record_id) is True
    assert len(store) == 0
    assert store.delete(record_id) is False


def test_clear_empties_the_store(store: BaseVectorStore):
    store.add("a", [1.0, 0.0])
    store.add("b", [0.0, 1.0])
    store.clear()
    assert len(store) == 0
    assert store.all() == []
    assert store.search([1.0, 0.0], k=5) == []


def test_clear_then_add_again_works(store: BaseVectorStore):
    """Guards the rebuild() pattern DenseRetriever relies on: clear the
    store, then re-add from the authoritative source of truth."""
    store.add("stale", [1.0, 0.0], record_id="x")
    store.clear()
    store.add("fresh", [1.0, 0.0], record_id="x")
    assert len(store) == 1
    hits = store.search([1.0, 0.0], k=1)
    assert hits[0][1] == "fresh"


def test_all_lists_records_without_exposing_embeddings(store: BaseVectorStore):
    store.add("hello", [1.0, 0.0], metadata={"tag": "greeting"})
    [record] = store.all()
    assert record["text"] == "hello"
    assert record["metadata"] == {"tag": "greeting"}
    assert "embedding" not in record
