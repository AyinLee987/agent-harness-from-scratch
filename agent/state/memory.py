"""Layered memory: short-term context management + long-term vector recall.

* :class:`ShortTermMemory` keeps a sliding window of recent messages and falls
  back to summarization when the window would blow the token budget.
* :class:`LongTermMemory` is a persistent-capable vector store. It defaults
  to an in-memory NumPy backend and accepts any :class:`~agent.vector_store.BaseVectorStore`
  implementation — e.g. :class:`~agent.vector_store.SQLiteVectorStore` for
  persistence, or a future FAISS/Qdrant backend.

Both are intentionally simple; the README's design notes discuss what you'd
swap in to scale them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..llm import BaseLLM, estimate_tokens
from ..memory.embeddings import EmbeddingProvider, LLMEmbeddingProvider
from .store import BaseVectorStore, NumPyVectorStore


def _tool_call_groups(messages: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Partition ``messages`` into atomic units for windowing.

    An assistant message carrying ``tool_calls`` plus every ``tool`` message
    that immediately follows it (its responses) must always move together —
    an OpenAI-compatible API rejects a ``tool`` message whose triggering
    ``tool_calls`` message isn't present. Every other message is its own
    one-message group.
    """

    groups: List[List[Dict[str, Any]]] = []
    i = 0
    n = len(messages)
    while i < n:
        message = messages[i]
        if message.get("role") == "assistant" and message.get("tool_calls"):
            j = i + 1
            while j < n and messages[j].get("role") == "tool":
                j += 1
            groups.append(messages[i:j])
            i = j
        else:
            groups.append([message])
            i += 1
    return groups


class ShortTermMemory:
    """Sliding-window message memory with a summarization fallback.

    Keeps the system message (if any) plus the most recent ``window`` messages.
    When the retained transcript exceeds ``max_tokens``, older messages are
    compressed into a single summary message via the LLM.
    """

    def __init__(
        self,
        llm: BaseLLM,
        window: int = 12,
        max_tokens: int = 4000,
    ) -> None:
        self._llm = llm
        self.window = window
        self.max_tokens = max_tokens

    def manage(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return a (possibly compressed) view of ``messages`` for the next call."""

        if self._token_count(messages) <= self.max_tokens and len(messages) <= self.window:
            return messages

        system = messages[0:1] if messages and messages[0].get("role") == "system" else []
        body = messages[len(system):]

        # Keep the freshest `window` messages verbatim, in whole tool-call
        # groups (see `_tool_call_groups`) so an assistant message that
        # declared tool_calls is never separated from its tool responses —
        # summarizing one half of such a group away produces a message list
        # the model API will reject as malformed. This means the kept
        # portion can run a little over `window` when the boundary group is
        # itself large; that's the correct trade-off over a well-formed cap.
        groups = _tool_call_groups(body)
        recent_groups: List[List[Dict[str, Any]]] = []
        kept = 0
        for group in reversed(groups):
            recent_groups.insert(0, group)
            kept += len(group)
            if kept >= self.window:
                break
        recent = [msg for group in recent_groups for msg in group]
        older = [
            msg
            for group in groups[: len(groups) - len(recent_groups)]
            for msg in group
        ]
        if not older:
            return system + recent

        summary = self._summarize(older)
        summary_msg = {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{summary}",
        }
        return system + [summary_msg] + recent

    def _summarize(self, messages: List[Dict[str, Any]]) -> str:
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in messages if m.get("content")
        )
        prompt = [
            {
                "role": "system",
                "content": "Summarize the conversation below concisely, "
                "preserving facts, decisions, and tool results.",
            },
            {"role": "user", "content": transcript},
        ]
        resp = self._llm.chat(prompt, tools=[])
        return (resp.content or transcript)[:2000]

    @staticmethod
    def _token_count(messages: List[Dict[str, Any]]) -> int:
        return sum(estimate_tokens(str(m.get("content") or "")) for m in messages)


@dataclass
class MemoryRecord:
    """A single long-term memory entry."""

    id: str = ""
    text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


class LongTermMemory:
    """Persistent-capable long-term memory with pluggable vector storage.

    Defaults to an in-memory :class:`~agent.vector_store.NumPyVectorStore`.
    Pass a :class:`~agent.vector_store.SQLiteVectorStore` (or any
    :class:`~agent.vector_store.BaseVectorStore` implementation) to
    ``vector_store`` for persistence.

    .. code-block:: python

        from agent import LongTermMemory, SQLiteVectorStore

        # In-memory (default, backward-compatible):
        mem = LongTermMemory(llm)

        # Persistent SQLite:
        store = SQLiteVectorStore("memory/knowledge.db")
        mem = LongTermMemory(llm, vector_store=store)

    The public interface (``add``, ``search``, ``__len__``) is unchanged
    regardless of the backend.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        vector_store: Optional[BaseVectorStore] = None,
        *,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        if embedding_provider is None:
            if llm is None:
                raise ValueError("LongTermMemory requires an embedding_provider.")
            embedding_provider = LLMEmbeddingProvider(llm)
        self._llm = llm  # backward-compatible attribute; no longer used directly
        self._embedding_provider = embedding_provider
        self._store = vector_store if vector_store is not None else NumPyVectorStore()

    @property
    def store(self) -> BaseVectorStore:
        """The underlying vector store (for direct access to ``all()``, etc.)."""
        return self._store

    def __len__(self) -> int:
        return len(self._store)

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Embed ``text`` and persist it. Returns the record id."""
        embedding = self._embedding_provider.embed_documents([text])[0]
        return self._store.add(text, embedding, metadata)

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        """Return up to ``k`` ``(text, score)`` pairs ranked by cosine similarity."""
        embedding = self._embedding_provider.embed_query(query)
        results = self._store.search(embedding, k)
        # Convert (id, text, score) -> (text, score) for backward compatibility.
        return [(text, score) for _, text, score in results]

    def as_search_tool(self):
        """Return a :class:`~agent.tools.FunctionTool` wrapping ``search()``.

        Register this with the agent's :class:`ToolRegistry` so the model can
        **decide** when to query long-term memory, rather than having it injected
        automatically before every run.

        .. code-block:: python

            mem = LongTermMemory(llm)
            mem.add("The capital of France is Paris.")

            registry = ToolRegistry([calculator, web_search])
            registry.register(mem.as_search_tool())  # ← RAG as a tool

            agent = ReActAgent(llm=llm, tools=registry, long_term=mem)
        """

        memory = self  # capture for closure

        def _search(query: str) -> str:
            """Search the agent's long-term memory for relevant facts and past interactions.

            Use this tool when you need to recall information that may have been
            stored from previous conversations or injected as domain knowledge.

            Args:
                query: Natural-language search query.
            """
            results = memory.search(query, k=3)
            if not results:
                return "No relevant memories found."
            return "\n".join(
                f"- {text} (relevance: {score:.2f})"
                for text, score in results
            )

        from ..tools import FunctionTool

        return FunctionTool(_search, name="memory_search")

    def list_all(self) -> List[MemoryRecord]:
        """Return all stored records."""
        return [
            MemoryRecord(
                id=r["id"],
                text=r["text"],
                metadata=r.get("metadata", {}),
                created_at=r.get("created_at", ""),
            )
            for r in self._store.all()
        ]

    def delete(self, record_id: str) -> bool:
        """Remove a record by id."""
        return self._store.delete(record_id)
