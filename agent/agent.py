"""Public API — thin facade that wires trigger + state + shared infrastructure.

The :class:`ReActAgent` class is now a lightweight wrapper around
:class:`~agent.trigger.ReActLoop`.  It exists for backward compatibility
and as a convenient single-import entry point.

Example::

    from agent import MockLLM, ReActAgent, ToolRegistry, tool

    @tool
    def calculator(expression: str) -> str:
        '''Evaluate an arithmetic expression.'''
        return str(eval(expression))

    agent = ReActAgent(llm=MockLLM(), tools=ToolRegistry([calculator]))
    print(agent.run("What is 23 times 17?").answer)
"""

from __future__ import annotations

import threading
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Sequence

from .compression import ContextCompressor
from .context import ContextProvider
from .llm import BaseLLM
from .memory import MemoryManager
from .safety import ToolOutputGuard
from .state.memory import LongTermMemory, ShortTermMemory
from .tools import ToolRegistry
from .trigger.events import RunEvent
from .trigger.react_loop import DEFAULT_SYSTEM_PROMPT, AgentResult, ReActLoop
from .trigger.tool_router import ToolSelector


class ReActAgent:
    """Thin facade around :class:`~agent.trigger.ReActLoop`.

    All parameters are forwarded directly to :class:`ReActLoop`.
    The ``run(task)`` method returns an :class:`AgentResult`.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        max_tokens: int = 100_000,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        compressor: Optional[ContextCompressor] = None,
        output_guard: Optional[ToolOutputGuard] = None,
        compress_at_fraction: float = 0.6,
        max_tool_retries: int = 1,
        loop_detection: bool = True,
        loop_same_call_limit: int = 3,
        source_failure_hint_threshold: int = 2,
        agent_name: str = "agent",
        memory_manager: Optional[MemoryManager] = None,
        memory_namespace: str = "default",
        memory_subject_id: str = "anonymous",
        context_providers: Optional[Sequence[ContextProvider]] = None,
        tool_selector: Optional[ToolSelector] = None,
    ) -> None:
        self._loop = ReActLoop(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_steps=max_steps,
            max_tokens=max_tokens,
            short_term=short_term,
            long_term=long_term,
            compressor=compressor,
            output_guard=output_guard,
            compress_at_fraction=compress_at_fraction,
            max_tool_retries=max_tool_retries,
            loop_detection=loop_detection,
            loop_same_call_limit=loop_same_call_limit,
            source_failure_hint_threshold=source_failure_hint_threshold,
            agent_name=agent_name,
            memory_manager=memory_manager,
            memory_namespace=memory_namespace,
            memory_subject_id=memory_subject_id,
            context_providers=context_providers,
            tool_selector=tool_selector,
        )

    def run(
        self,
        task: str,
        cancellation_event: Optional[threading.Event] = None,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Run the agent to completion on ``task``.

        ``resume_from`` continues a run that suspended on a long-running
        job -- see :meth:`agent.trigger.ReActLoop.run`.
        """
        return self._loop.run(
            task, cancellation_event=cancellation_event, resume_from=resume_from
        )

    def iter_run(
        self,
        task: str,
        cancellation_event: Optional[threading.Event] = None,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> Iterator[RunEvent]:
        """Run ``task``, yielding a :class:`RunEvent` as each thing happens.

        The generator's return value is the :class:`AgentResult`. See
        :meth:`agent.trigger.ReActLoop.iter_run`.
        """
        return self._loop.iter_run(
            task, cancellation_event=cancellation_event, resume_from=resume_from
        )

    def aiter_run(
        self,
        task: str,
        cancellation_event: Optional[threading.Event] = None,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[RunEvent]:
        """:meth:`iter_run` for a caller that owns an event loop.

        The final :class:`AgentResult` arrives on the terminating
        ``run_completed`` event's ``result`` key, since an async generator
        cannot return a value. See :meth:`agent.trigger.ReActLoop.aiter_run`.
        """
        return self._loop.aiter_run(
            task, cancellation_event=cancellation_event, resume_from=resume_from
        )


# -- delegated attributes ---------------------------------------------------
# These used to be plain copies assigned in ``__init__`` and labelled
# "exposed for backward compatibility". That made them true in one direction
# only: reading worked, but *writing* one changed just the copy while
# ``ReActLoop`` went on reading its own -- silently, with no error and no
# warning. ``AgentGateway.run()``'s per-request ``agent.max_steps = n``
# override was doing exactly that and had no effect at all. Properties keep
# the identical read API and make writes actually land. See BUGS.md #14.
_DELEGATED = (
    "llm",
    "tools",
    "system_prompt",
    "max_steps",
    "max_tokens",
    "short_term",
    "long_term",
    "compressor",
    "output_guard",
    "agent_name",
    "memory_manager",
    "memory_namespace",
    "memory_subject_id",
    "context_providers",
    "tool_selector",
)


def _delegate(name: str) -> property:
    return property(
        lambda self: getattr(self._loop, name),
        lambda self, value: setattr(self._loop, name, value),
        doc=f"Delegates to :attr:`~agent.trigger.ReActLoop.{name}`.",
    )


for _attribute in _DELEGATED:
    setattr(ReActAgent, _attribute, _delegate(_attribute))
del _attribute
