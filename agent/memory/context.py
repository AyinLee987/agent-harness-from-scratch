"""Threads a stored multi-turn conversation into an otherwise stateless run.

``ReActLoop.run()`` builds a brand-new ``ExecutionContext`` on every call —
nothing from a prior call carries over (see ``agent/trigger/react_loop.py``).
Continuing a conversation across separate ``run()`` calls (e.g. separate HTTP
requests) is therefore an application-level concern, not something the loop
does for you. This module is that missing piece: it replays a session's
stored history through the loop's existing ``ContextProvider`` hook, so a
stateless ``run()`` call can still see "everything said so far" without
``agent.py`` or ``react_loop.py`` needing to know sessions exist.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from ..llm import BaseLLM
from .models import SummarySnapshot
from .session import SessionMemoryStore

_SUMMARIZE_INSTRUCTION = (
    "Summarize the conversation below concisely, preserving facts, decisions, "
    "and tool results. This will replace the messages for future turns, so do "
    "not omit anything a later turn might need to refer back to."
)


class SessionContextProvider:
    """Implements :class:`~agent.context.ContextProvider` over a
    :class:`~agent.memory.session.SessionMemoryStore`.

    One instance is scoped to a single ``conversation_id``. Register it via
    ``ReActAgent(context_providers=[...])`` and call :meth:`record_turn` after
    each ``run()`` call completes — the provider does not do this itself, the
    same way ``ExecutionContext`` does not decide when a run is over.

    Only the user message and the final assistant answer are persisted per
    turn, not the intermediate tool-call/tool-result messages a run produces
    internally — a later turn replays this as plain conversation, and the
    tools available then may not even be the same ones. This mirrors what a
    human would remember from the exchange, not a raw execution log; the raw
    log is what ``AgentResult.trajectory`` is for.

    Once a conversation grows past ``summarize_beyond`` stored messages, older
    messages are folded into a single summary (persisted via
    ``store.save_summary``) instead of being replayed verbatim every turn.
    Without this, every new ``run()`` call would reload the full history and
    still pay for :class:`~agent.state.memory.ShortTermMemory` to
    re-summarize it from scratch on top — an ever-growing, uncached cost,
    since that summary is scoped to a single run and is never persisted.
    Summarization needs an ``llm``; without one, history is replayed verbatim
    regardless of length (correct for short conversations, unbounded for
    long ones — pass ``llm`` once conversations are expected to run long).
    """

    def __init__(
        self,
        store: SessionMemoryStore,
        conversation_id: str,
        *,
        llm: Optional[BaseLLM] = None,
        recent_window: int = 12,
        summarize_beyond: int = 24,
    ) -> None:
        self.store = store
        self.conversation_id = conversation_id
        self.llm = llm
        self.recent_window = recent_window
        self.summarize_beyond = summarize_beyond

    def prepare(self, task: str) -> Sequence[Dict[str, Any]]:
        """Return this conversation's history as messages to inject before ``task``."""

        messages = self.store.load_messages(self.conversation_id)
        if self.llm is None or len(messages) <= self.summarize_beyond:
            return messages

        recent = messages[-self.recent_window :]
        older = messages[: len(messages) - len(recent)]
        summary_message = self._summary_message(older)
        return ([summary_message] if summary_message else []) + recent

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        """Persist one completed turn. Call this after every ``run()``."""

        self.store.append_message(
            self.conversation_id, {"role": "user", "content": user_message}
        )
        self.store.append_message(
            self.conversation_id, {"role": "assistant", "content": assistant_message}
        )

    def _summary_message(self, older: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not older:
            snapshot = self.store.load_summary(self.conversation_id)
            return self._render(snapshot) if snapshot else None

        newest_id = older[-1].get("id")
        snapshot = self.store.load_summary(self.conversation_id)
        if snapshot is not None and snapshot.through_message_id == newest_id:
            return self._render(snapshot)

        summary_text = self._summarize(older, prior=snapshot.summary if snapshot else "")
        snapshot = SummarySnapshot(
            conversation_id=self.conversation_id,
            summary=summary_text,
            through_message_id=newest_id,
        )
        self.store.save_summary(snapshot)
        return self._render(snapshot)

    def _summarize(self, messages: List[Dict[str, Any]], *, prior: str) -> str:
        assert self.llm is not None
        transcript = "\n".join(
            f"{m.get('role')}: {m.get('content')}" for m in messages if m.get("content")
        )
        body = (
            f"Prior summary:\n{prior}\n\nNew messages since then:\n{transcript}"
            if prior
            else transcript
        )
        prompt = [
            {"role": "system", "content": _SUMMARIZE_INSTRUCTION},
            {"role": "user", "content": body},
        ]
        resp = self.llm.chat(prompt, tools=[])
        return (resp.content or transcript)[:2000]

    @staticmethod
    def _render(snapshot: SummarySnapshot) -> Dict[str, Any]:
        return {
            "role": "system",
            "content": f"Summary of earlier conversation:\n{snapshot.summary}",
        }
