"""Execution context: the mutable state threaded through an agent run.

The :class:`ExecutionContext` is deliberately the *only* object that owns
mutable run state. Everything the loop needs to make a decision -- the message
transcript, scratch state, the step-by-step trajectory, and the token/step
budget -- lives here. Keeping it in one place makes runs easy to log, replay,
and reason about, and it is where the budget guardrail is enforced.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """One iteration of the ReAct loop: thought -> action -> observation.

    A model turn can request **several** tool calls at once, so a step is
    not one call. :attr:`tool_calls` records each one separately; ``action``
    and ``observation`` remain the flattened view (the first call, and every
    observation joined) that the UI, checkpoints and loop detection have
    always read.

    Keeping only the flattened view lost information that nothing else could
    recover: a turn that called ``fetch`` then ``summarize`` recorded just
    ``fetch``, and if ``summarize`` failed, its error text was concatenated
    onto ``fetch``'s success -- so trajectory metrics, the Worker's
    ``tool_call_summary``, and eval's ``expect_tool`` check all reported the
    step as one successful ``fetch``. See BUGS.md #16.
    """

    index: int
    id: str = ""  # correlation id: "{run_id}-{index}"
    thought: str = ""
    action: Optional[Dict[str, Any]] = None  # {"name": str, "arguments": dict}
    observation: Optional[str] = None
    error: Optional[str] = None
    #: One entry per tool call in this step, in the order the model issued
    #: them: {"id", "name", "arguments", "observation", "ok"}.
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def record_tool_call(
        self,
        *,
        id: str,
        name: str,
        arguments: Any,
        observation: Optional[str],
        ok: bool,
    ) -> None:
        """Append one call's outcome, and keep the flattened view in sync."""

        self.tool_calls.append(
            {
                "id": id,
                "name": name,
                "arguments": arguments,
                "observation": observation,
                "ok": ok,
            }
        )
        if self.action is None:
            self.action = {"name": name, "arguments": arguments}
        observations = [
            str(call["observation"])
            for call in self.tool_calls
            if call["observation"] is not None
        ]
        self.observation = "\n".join(observations) if observations else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "id": self.id,
            "thought": self.thought,
            "action": self.action,
            "observation": self.observation,
            "error": self.error,
            "tool_calls": list(self.tool_calls),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step":
        """Rebuild from :meth:`to_dict`, tolerating missing keys.

        A checkpoint written before ``tool_calls`` existed has none; the
        flattened ``action`` is reconstructed into a single entry so
        consumers can read one shape regardless of when it was written.
        """

        recorded = list(data.get("tool_calls") or [])
        action = data.get("action")
        observation = data.get("observation")
        if not recorded and action:
            recorded = [
                {
                    "id": "",
                    "name": action.get("name"),
                    "arguments": action.get("arguments"),
                    "observation": observation,
                    "ok": not str(observation or "").startswith("ERROR"),
                }
            ]
        return cls(
            index=int(data.get("index", 0)),
            id=str(data.get("id") or ""),
            thought=str(data.get("thought") or ""),
            action=action,
            observation=observation,
            error=data.get("error"),
            tool_calls=recorded,
        )


@dataclass
class ExecutionContext:
    """Owns all mutable state for a single agent run.

    Attributes:
        run_id: Short correlation id for this run; every step id derives from it,
            so logs/traces/observations can be tied back to a single run.
        messages: The running chat transcript (OpenAI message dicts).
        state: Free-form scratch space for tools/agents to stash data.
        steps: The recorded trajectory, one :class:`Step` per loop iteration.
        max_steps: Hard cap on loop iterations (anti-runaway guardrail).
        max_tokens: Hard cap on cumulative tokens (anti-runaway guardrail).
        tokens_used: Running total of tokens consumed across LLM calls.
    """

    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    messages: List[Dict[str, Any]] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    steps: List[Step] = field(default_factory=list)
    max_steps: int = 10
    max_tokens: int = 100_000
    tokens_used: int = 0

    # -- message helpers --------------------------------------------------
    def add_message(self, role: str, content: Any, **extra: Any) -> Dict[str, Any]:
        """Append a message to the transcript and return it."""

        message: Dict[str, Any] = {"role": role, "content": content}
        message.update(extra)
        self.messages.append(message)
        return message

    # -- step / trajectory helpers ---------------------------------------
    def new_step(self) -> Step:
        """Create, register, and return the next :class:`Step`."""

        index = len(self.steps)
        step = Step(index=index, id=f"{self.run_id}-{index}")
        self.steps.append(step)
        return step

    @property
    def current_step(self) -> Optional[Step]:
        return self.steps[-1] if self.steps else None

    # -- budget guardrail -------------------------------------------------
    def add_tokens(self, count: int) -> None:
        self.tokens_used += max(0, count)

    def over_budget(self) -> bool:
        """True if either the step or token budget has been exhausted."""

        return len(self.steps) >= self.max_steps or self.tokens_used >= self.max_tokens

    def budget_reason(self) -> Optional[str]:
        """Human-readable reason the budget tripped, or ``None``."""

        if len(self.steps) >= self.max_steps:
            return f"max_steps ({self.max_steps}) reached"
        if self.tokens_used >= self.max_tokens:
            return f"max_tokens ({self.max_tokens}) reached"
        return None

    # -- serialization ----------------------------------------------------
    def trajectory(self) -> List[Dict[str, Any]]:
        """Return the recorded steps as plain dicts (for logging/eval)."""

        return [s.to_dict() for s in self.steps]

    def checkpoint(self) -> Dict[str, Any]:
        """A complete, JSON-serializable snapshot of this run.

        Everything the loop needs to pick the run back up: the transcript,
        the scratch state, the trajectory, and the budget already consumed.
        This is what makes a run that suspends on a long-running job (see
        :mod:`agent.jobs`) resumable rather than lost -- without it, an
        agent's whole state lives only in one process's heap, and a
        half-hour tool call is a half-hour window in which a restart throws
        the run away.

        ``state`` can hold arbitrary tool scratch data, so any value that
        will not serialize is dropped rather than failing the checkpoint --
        a run that cannot be snapshotted at all is strictly worse than one
        snapshotted without a cache entry some tool stashed.
        """

        return {
            "run_id": self.run_id,
            "messages": self.messages,
            "state": {
                key: value
                for key, value in self.state.items()
                if _json_safe(value)
            },
            "steps": self.trajectory(),
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
            "tokens_used": self.tokens_used,
        }

    @classmethod
    def restore(cls, data: Dict[str, Any]) -> "ExecutionContext":
        """Rebuild a context from :meth:`checkpoint`.

        Version-tolerant: a checkpoint written by an older build must still
        resume rather than crash, so every field falls back to its default.
        """

        ctx = cls(
            run_id=str(data.get("run_id") or uuid.uuid4().hex[:8]),
            messages=list(data.get("messages") or []),
            state=dict(data.get("state") or {}),
            max_steps=int(data.get("max_steps", 10)),
            max_tokens=int(data.get("max_tokens", 100_000)),
            tokens_used=int(data.get("tokens_used", 0)),
        )
        ctx.steps = [Step.from_dict(item) for item in (data.get("steps") or [])]
        return ctx


def _json_safe(value: Any) -> bool:
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
