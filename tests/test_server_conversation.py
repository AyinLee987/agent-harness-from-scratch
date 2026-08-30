"""Server-level checks for conversation continuity on /api/run.

See agent/memory/context.py for why ReActLoop.run() needs help remembering
anything across separate calls at all.
"""

from __future__ import annotations

import asyncio

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
