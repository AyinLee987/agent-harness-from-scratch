"""What a run emits, and what a run asks for.

Two small vocabularies that together let one implementation of the ReAct
state machine serve both a synchronous caller and a streaming HTTP endpoint.

**Events** (:class:`RunEvent`) are what a run *tells* you: it started, the
model was called, a tool ran, it finished. ``ReActLoop.iter_run()`` yields
them; ``run()`` drains them and returns the final result, while
``/api/stream`` forwards them as Server-Sent Events. Before this existed the
streaming endpoint re-implemented the whole loop inline to get at those
moments, and the two copies drifted -- see BUGS.md #22.

**Effects** (:class:`CallModel`, :class:`CallTool`) are what a run *asks
for*. Graph nodes do not call the provider or the tool themselves; they
``yield`` a description of the call and are handed the result back. That is
what makes one node body drivable by both a blocking runner and an
``await``-ing one: a ``def`` cannot ``await``, but it can ``yield`` and let
whoever is driving it decide how the call gets made.

Everything here is deliberately data, not behaviour. Neither vocabulary
knows about HTTP, SSE, asyncio, or any provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# -- event kinds ------------------------------------------------------------
#: The run began. ``run_id``, ``task``.
RUN_STARTED = "run_started"
#: A think step is about to call the model. ``step``.
THINK_STARTED = "think_started"
#: One token (or chunk) of the model's reply. ``step``, ``token``. Emitted
#: only when the run is driven asynchronously against a streaming provider;
#: a blocking ``chat()`` has nothing to emit until it returns.
TEXT = "text"
#: The model replied. ``step``, ``thought``, ``tool_calls``, ``usage``.
THINK_COMPLETED = "think_completed"
#: A forced-reflection turn produced plain text. ``step``, ``text``.
REFLECTION = "reflection"
#: A tool call is about to be dispatched. ``step``, ``id``, ``name``,
#: ``arguments``.
TOOL_STARTED = "tool_started"
#: A tool call returned. ``step``, ``id``, ``name``, ``observation``, ``ok``.
#: Pairs one-to-one with :data:`TOOL_STARTED`, and with the per-call records
#: on ``Step.tool_calls`` (BUGS.md #16).
TOOL_COMPLETED = "tool_completed"
#: The run is waiting on long-running jobs and has a resumable checkpoint.
#: ``pending_job_ids``. Not a failure -- see ``agent/jobs/``.
SUSPENDED = "suspended"
#: Something went wrong. ``step``, ``type``, ``message``.
ERROR = "error"
#: The run is over. ``answer``, ``stop_reason``, ``steps``, ``tokens``.
RUN_COMPLETED = "run_completed"


@dataclass(frozen=True)
class RunEvent:
    """One thing that happened during a run.

    ``kind`` is one of the constants above; ``data`` is a plain
    JSON-serializable dict, because the main consumer serializes it straight
    onto the wire.
    """

    kind: str
    data: Dict[str, Any] = field(default_factory=dict)


# -- effects ----------------------------------------------------------------
@dataclass(frozen=True)
class CallModel:
    """A node asking for one model call.

    The node has already done all the pure work -- window management,
    compression, tool selection, budget projection -- so this carries the
    finished prompt and nothing else. The driver decides whether that means
    ``llm.chat()`` on this thread, ``chat()`` in a worker thread, or
    ``llm.astream()`` with token events along the way.

    ``step_index`` rides along only so streamed :data:`TEXT` events can be
    attributed to the right step.
    """

    messages: List[Dict[str, Any]]
    tools: Optional[List[Dict[str, Any]]]
    step_index: int


@dataclass(frozen=True)
class ManageContext:
    """A node asking for the prompt to be trimmed and compressed.

    Looks like bookkeeping and is not: ``ShortTermMemory.manage()``
    summarizes the overflow with a **model call**, and
    ``ContextCompressor`` can too. Leaving it in the node body would leave
    the async driver with a blocking provider call it could not see -- the
    exact failure the effect boundary exists to prevent, just in the one
    place it is easy to forget.

    Carries no arguments: everything it needs is on the loop and its
    context, and the driver calls one method.
    """


@dataclass(frozen=True)
class CallTool:
    """A node asking for one tool call.

    Dispatch is synchronous and can block for a long time (an MCP round
    trip, a subagent, a shell command), which is exactly why the node does
    not perform it: the async driver runs it in a worker thread so a
    streaming request cannot stall the event loop for every other request in
    the process.
    """

    name: str
    arguments: Any


#: Anything a node may yield expecting a value back. A yielded
#: :class:`RunEvent` is fire-and-forget and is *not* an effect.
Effect = (ManageContext, CallModel, CallTool)
