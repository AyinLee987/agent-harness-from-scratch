"""Tests for SessionContextProvider — the ContextProvider that threads a
SessionMemoryStore's history into an otherwise stateless ReActLoop run.
"""

from __future__ import annotations

from agent import (
    InMemorySessionStore,
    LLMResponse,
    MockLLM,
    ReActAgent,
    SessionContextProvider,
    ToolRegistry,
    Usage,
)


class _EchoingSummaryLLM(MockLLM):
    """Returns a fixed, recognizable summary instead of calling a real model."""

    def chat(self, messages, tools=None):
        if not tools:
            return LLMResponse(content="SUMMARY", usage=Usage(1, 1))
        return super().chat(messages, tools=tools)


def test_prepare_replays_all_stored_messages_without_an_llm():
    store = InMemorySessionStore()
    provider = SessionContextProvider(store, "conv-1")
    store.append_message("conv-1", {"role": "user", "content": "first"})
    store.append_message("conv-1", {"role": "assistant", "content": "first reply"})

    prepared = provider.prepare("second")

    assert [m["content"] for m in prepared] == ["first", "first reply"]


def test_record_turn_persists_only_user_and_assistant_messages():
    store = InMemorySessionStore()
    provider = SessionContextProvider(store, "conv-1")

    provider.record_turn("what is 2+2?", "4")

    stored = store.load_messages("conv-1")
    assert [(m["role"], m["content"]) for m in stored] == [
        ("user", "what is 2+2?"),
        ("assistant", "4"),
    ]


def test_each_conversation_id_is_isolated():
    store = InMemorySessionStore()
    SessionContextProvider(store, "conv-a").record_turn("a-question", "a-answer")
    SessionContextProvider(store, "conv-b").record_turn("b-question", "b-answer")

    assert [m["content"] for m in store.load_messages("conv-a")] == ["a-question", "a-answer"]
    assert [m["content"] for m in store.load_messages("conv-b")] == ["b-question", "b-answer"]


def test_history_survives_across_separate_agent_instances():
    """The scenario this module exists for: two separate run() calls, on two
    separate ReActAgent instances (standing in for two separate HTTP
    requests), still see the earlier turn."""

    store = InMemorySessionStore()
    provider = SessionContextProvider(store, "conv-1")

    seen_by_second_call = {}

    class _RecordingLLM(MockLLM):
        def chat(self, messages, tools=None):
            seen_by_second_call["messages"] = messages
            return super().chat(messages, tools=tools)

    provider.record_turn("my name is Zhuoyang", "noted")

    agent = ReActAgent(llm=_RecordingLLM(), tools=ToolRegistry(), context_providers=[provider])
    agent.run("what is my name?")

    contents = [m["content"] for m in seen_by_second_call["messages"]]
    assert "my name is Zhuoyang" in contents


def test_short_history_is_not_summarized():
    store = InMemorySessionStore()
    for i in range(4):
        store.append_message("conv-1", {"role": "user", "content": f"m{i}"})
    provider = SessionContextProvider(
        store, "conv-1", llm=_EchoingSummaryLLM(), recent_window=4, summarize_beyond=6
    )

    prepared = provider.prepare("next")

    assert [m["content"] for m in prepared] == ["m0", "m1", "m2", "m3"]
    assert store.load_summary("conv-1") is None


def test_long_history_is_folded_into_a_persisted_summary():
    store = InMemorySessionStore()
    for i in range(10):
        store.append_message("conv-1", {"role": "user", "content": f"m{i}"})
    provider = SessionContextProvider(
        store, "conv-1", llm=_EchoingSummaryLLM(), recent_window=4, summarize_beyond=6
    )

    prepared = provider.prepare("next")

    assert prepared[0]["role"] == "system"
    assert prepared[0]["content"] == "Summary of earlier conversation:\nSUMMARY"
    assert [m["content"] for m in prepared[1:]] == ["m6", "m7", "m8", "m9"]

    snapshot = store.load_summary("conv-1")
    assert snapshot is not None
    assert snapshot.through_message_id is not None


def test_summary_is_reused_when_no_new_older_messages_arrived():
    store = InMemorySessionStore()
    calls = {"count": 0}

    class _CountingSummaryLLM(_EchoingSummaryLLM):
        def chat(self, messages, tools=None):
            if not tools:
                calls["count"] += 1
            return super().chat(messages, tools=tools)

    for i in range(10):
        store.append_message("conv-1", {"role": "user", "content": f"m{i}"})
    provider = SessionContextProvider(
        store, "conv-1", llm=_CountingSummaryLLM(), recent_window=4, summarize_beyond=6
    )

    provider.prepare("first call")
    provider.prepare("second call, nothing new")

    assert calls["count"] == 1


def test_summary_advances_once_enough_new_messages_accumulate():
    store = InMemorySessionStore()

    class _VersionedSummaryLLM(MockLLM):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def chat(self, messages, tools=None):
            if not tools:
                self.n += 1
                return LLMResponse(content=f"SUMMARY-{self.n}", usage=Usage(1, 1))
            return super().chat(messages, tools=tools)

    for i in range(10):
        store.append_message("conv-1", {"role": "user", "content": f"m{i}"})
    provider = SessionContextProvider(
        store, "conv-1", llm=_VersionedSummaryLLM(), recent_window=4, summarize_beyond=6
    )
    first = provider.prepare("t1")

    for i in range(10, 14):
        store.append_message("conv-1", {"role": "user", "content": f"m{i}"})
    second = provider.prepare("t2")

    assert first[0]["content"] != second[0]["content"]
