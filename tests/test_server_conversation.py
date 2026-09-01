"""Server-level checks for conversation continuity on /api/run, /api/stream,
and the conversation-title endpoint.

See agent/memory/context.py for why ReActLoop.run() needs help remembering
anything across separate calls at all.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from app import server
from agent import LLMResponse, MockLLM, Usage


class _NameRememberingLLM(MockLLM):
    """Answers based on whether "Zhuoyang" appears anywhere in the messages
    it's shown — a stand-in for "does this call actually see prior turns."
    """

    def chat(self, messages, tools=None):
        contents = [str(m.get("content") or "") for m in messages]
        if any("Zhuoyang" in c for c in contents):
            return LLMResponse(content="Your name is Zhuoyang.", usage=Usage(1, 1))
        return LLMResponse(content="I don't know your name.", usage=Usage(1, 1))


def test_conversation_id_threads_history_across_separate_run_calls(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    first = asyncio.run(
        server.run(server.RunRequest(task="My name is Zhuoyang.", conversation_id="conv-1"))
    )
    second = asyncio.run(
        server.run(server.RunRequest(task="What is my name?", conversation_id="conv-1"))
    )

    assert first.conversation_id == "conv-1"
    assert second.conversation_id == "conv-1"
    assert second.answer == "Your name is Zhuoyang."


def test_omitting_conversation_id_stays_fully_stateless(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    asyncio.run(server.run(server.RunRequest(task="My name is Zhuoyang.")))
    second = asyncio.run(server.run(server.RunRequest(task="What is my name?")))

    assert second.conversation_id is None
    assert second.answer == "I don't know your name."


def test_different_conversation_ids_do_not_leak_into_each_other(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    asyncio.run(
        server.run(server.RunRequest(task="My name is Zhuoyang.", conversation_id="conv-a"))
    )
    other = asyncio.run(
        server.run(server.RunRequest(task="What is my name?", conversation_id="conv-b"))
    )

    assert other.answer == "I don't know your name."


def test_a_turn_persists_only_the_user_message_and_final_answer(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    asyncio.run(
        server.run(server.RunRequest(task="My name is Zhuoyang.", conversation_id="conv-2"))
    )

    stored = server.SESSION_STORE.load_messages("conv-2")
    assert [(m["role"], m["content"]) for m in stored] == [
        ("user", "My name is Zhuoyang."),
        ("assistant", "Your name is Zhuoyang."),
    ]


async def _run_stream(task, conversation_id=None):
    response = await server.stream(task, conversation_id=conversation_id)
    return "".join([chunk async for chunk in response.body_iterator])


def test_conversation_id_threads_history_on_stream_endpoint(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    first = asyncio.run(_run_stream("My name is Zhuoyang.", "conv-stream-1"))
    second = asyncio.run(_run_stream("What is my name?", "conv-stream-1"))

    assert '"conversation_id": "conv-stream-1"' in first
    assert '"text": "Your name is Zhuoyang."' in second


def test_stream_without_conversation_id_stays_stateless(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    asyncio.run(_run_stream("My name is Zhuoyang."))
    second = asyncio.run(_run_stream("What is my name?"))

    assert '"conversation_id": null' in second
    assert "I don't know your name." in second


def test_stream_persists_only_user_message_and_final_answer(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    asyncio.run(_run_stream("My name is Zhuoyang.", "conv-stream-2"))

    stored = server.SESSION_STORE.load_messages("conv-stream-2")
    assert [(m["role"], m["content"]) for m in stored] == [
        ("user", "My name is Zhuoyang."),
        ("assistant", "Your name is Zhuoyang."),
    ]


def test_stream_persists_the_user_message_even_if_the_client_disconnects_early(monkeypatch):
    """Regression test: a stream that never finishes iterating (the client
    refreshed/disconnected mid-run) must not lose the question — only the
    assistant's side is allowed to be missing. See agent/memory/context.py's
    record_user_message for why."""

    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)

    async def abandon_mid_stream():
        response = await server.stream(
            "My name is Zhuoyang.", conversation_id="conv-stream-disconnect"
        )
        agen = response.body_iterator
        await agen.__anext__()  # consume exactly one chunk, like an abandoned client
        await agen.aclose()  # simulate the disconnect tearing the generator down

    asyncio.run(abandon_mid_stream())

    stored = server.SESSION_STORE.load_messages("conv-stream-disconnect")
    assert [(m["role"], m["content"]) for m in stored] == [
        ("user", "My name is Zhuoyang."),
    ]


def test_conversation_messages_returns_empty_list_for_a_fresh_conversation_id():
    result = asyncio.run(server.conversation_messages("brand-new-conv"))

    assert result == {"conversation_id": "brand-new-conv", "messages": []}


def test_conversation_messages_returns_stored_turns(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", _NameRememberingLLM)
    asyncio.run(
        server.run(server.RunRequest(task="My name is Zhuoyang.", conversation_id="conv-msgs"))
    )

    result = asyncio.run(server.conversation_messages("conv-msgs"))

    assert result["conversation_id"] == "conv-msgs"
    assert [(m["role"], m["content"]) for m in result["messages"]] == [
        ("user", "My name is Zhuoyang."),
        ("assistant", "Your name is Zhuoyang."),
    ]


def test_conversation_title_404s_for_an_unknown_conversation():
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.conversation_title("does-not-exist"))
    assert exc_info.value.status_code == 404


def test_conversation_title_falls_back_to_first_message_under_mockllm(monkeypatch):
    monkeypatch.setattr(server, "_build_llm", MockLLM)
    server.SESSION_STORE.append_message("conv-title-1", {"role": "user", "content": "My name is Zhuoyang."})
    server.SESSION_STORE.append_message("conv-title-1", {"role": "assistant", "content": "noted"})

    result = asyncio.run(server.conversation_title("conv-title-1"))

    assert result["title"] == "My name is Zhuoyang."


def test_conversation_title_uses_the_llm_summary_when_one_is_available(monkeypatch):
    class _TitleLLM(MockLLM):
        def chat(self, messages, tools=None):
            return LLMResponse(content="Introduction and name check", usage=Usage(1, 1))

    monkeypatch.setattr(server, "_build_llm", _TitleLLM)
    server.SESSION_STORE.append_message("conv-title-2", {"role": "user", "content": "My name is Zhuoyang."})
    server.SESSION_STORE.append_message("conv-title-2", {"role": "assistant", "content": "noted"})

    result = asyncio.run(server.conversation_title("conv-title-2"))

    assert result["title"] == "Introduction and name check"
