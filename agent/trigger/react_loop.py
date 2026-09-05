"""ReAct loop — the think → act → observe cycle.

Extracted from :class:`~agent.ReActAgent`.  The :class:`ReActLoop` owns the
graph construction, the think/act node implementations, and the main ``run()``
entry point.  It is wired together with a :class:`StateGraph` from
:mod:`.graph` and a :class:`ToolDispatcher` from :mod:`.dispatch`.

Guardrails included: a hard step/token budget, a finish check, and
retry-once-then-fail handling of malformed tool calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple

from ..compression import ContextCompressor
from ..context import ContextProvider
from ..errors import FatalToolError
from ..jobs.models import SuspendRun
from ..llm import BaseLLM, LLMResponse, ToolCall, Usage, estimate_tokens
from ..memory import MemoryManager, RunCompletedEvent
from ..observability import bind_log_context, get_logger, log_event, run_log_file
from ..retry import TransientLLMError
from ..safety import ToolOutputGuard
from ..state.context import ExecutionContext, Step
from ..state.memory import LongTermMemory, ShortTermMemory
from ..tools import ToolRegistry
from .dispatch import ToolDispatcher, is_failure_observation
from .events import (
    ERROR,
    REFLECTION,
    RUN_COMPLETED,
    RUN_STARTED,
    SUSPENDED,
    TEXT,
    THINK_COMPLETED,
    THINK_STARTED,
    TOOL_COMPLETED,
    TOOL_STARTED,
    CallModel,
    CallTool,
    ManageContext,
    RunEvent,
)
from .graph import StateGraph
from .tool_router import ToolSelector, filtered_schemas

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful ReAct agent. Reason step by step. Use the provided tools "
    "when they help answer the user's request. If a memory_search tool is available "
    "and the user asks about something you might have stored from past conversations "
    "or domain knowledge, call it proactively. When you have enough information, "
    "respond with a final answer and do not call any more tools."
)

# Set on ctx.state after a step where at least one tool call failed (see
# is_failure_observation). The *next* think call is forced to run with no
# tool schemas offered, so the model must respond in plain text instead of
# immediately firing off another tool call -- a soft "try a different
# source" hint appended to the observation text turned out not to be enough
# on its own (a model mid-retry-spiral kept issuing tool calls straight
# through it); withholding tools for one turn makes reflection mandatory
# rather than hoped-for. See FORCED_REFLECTION_PROMPT and its use in
# _act_node / _think_node below.
REFLECT_AFTER_FAILURE_STATE_KEY = "__reflect_after_failure__"

FORCED_REFLECTION_PROMPT = (
    "Your last tool call failed. Before doing anything else, briefly explain "
    "(1) why you think it failed and (2) what you will try differently -- a "
    "different source or approach, or stopping here and reporting what you "
    "already have. Do not call a tool in this reply; you will get to act "
    "again right after."
)

logger = get_logger(__name__)


#: ``stop_reason`` for a run that stopped because it is waiting on
#: long-running jobs. Distinct from every other stop reason in that it is
#: not a failure and not an ending: the run is expected to continue, via
#: ``ReActLoop.run(resume_from=result.checkpoint)``.
SUSPENDED_STOP_REASON = "suspended_on_jobs"


@dataclass
class AgentResult:
    """The outcome of an agent run.

    ``checkpoint``/``pending_job_ids`` are populated only when
    ``stop_reason`` is :data:`SUSPENDED_STOP_REASON`; for every other
    outcome they stay ``None``/empty, so nothing that does not use jobs
    pays any attention to them.
    """

    answer: str
    success: bool
    steps: int
    tokens: int
    stop_reason: str
    trajectory: List[Dict[str, Any]]
    checkpoint: Optional[Dict[str, Any]] = None
    pending_job_ids: List[str] = field(default_factory=list)

    @property
    def suspended(self) -> bool:
        return self.stop_reason == SUSPENDED_STOP_REASON


class ReActLoop:
    """A minimal but production-shaped ReAct agent loop.

    The loop is powered by a :class:`~.graph.StateGraph` that encodes the ReAct
    state machine.  Nodes are ``think`` (LLM call) and ``act`` (tool dispatch).
    The graph is built once at construction time and reused across runs.
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
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.short_term = short_term or ShortTermMemory(llm)
        self.long_term = long_term
        self.compressor = compressor
        self.compress_at_fraction = compress_at_fraction
        self.output_guard = output_guard
        self.max_tool_retries = max_tool_retries
        self.loop_detection = loop_detection
        self.loop_same_call_limit = loop_same_call_limit
        self.agent_name = agent_name
        self.memory_manager = memory_manager
        self.memory_namespace = memory_namespace
        self.memory_subject_id = memory_subject_id
        self.context_providers = list(context_providers or [])
        # None keeps every tool's schema in every prompt -- the behaviour
        # before tool routing existed, and still the right one for a
        # handful of tools. See agent/trigger/tool_router.py.
        self.tool_selector = tool_selector
        self._dispatcher = ToolDispatcher(
            tools,
            max_retries=max_tool_retries,
            source_failure_hint_threshold=source_failure_hint_threshold,
        )
        self._graph = self._build_graph()

    # -- graph construction -------------------------------------------------
    def _build_graph(self) -> StateGraph:
        """Build the ReAct state graph.

        ::

            [think] ──(tool_calls)──▶ [act] ──(fixed)──▶ [think]
               │             │
               │             └──(forced reflection)──▶ [think]
               └──(answer)──▶ __end__

        The ``think`` → ``think`` self-edge is for a forced-reflection turn
        (see REFLECT_AFTER_FAILURE_STATE_KEY): that branch of _think_node
        never dispatches a tool and never finishes, it just needs to loop
        straight back to a normal think call.
        """
        g = StateGraph()
        g.add_node("think", self._think_node)
        g.add_node("act", self._act_node)
        g.set_entry_point("think")
        g.add_conditional_edges(
            "think",
            _route_by_next,
            {"tools": "act", "finish": "__end__", "think": "think"},
        )
        g.add_conditional_edges(
            "act",
            _route_by_next,
            {"think": "think", "finish": "__end__"},
        )
        return g

    # -- public API ---------------------------------------------------------
    def run(
        self,
        task: str,
        cancellation_event: Optional[threading.Event] = None,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """Run the agent to completion on ``task`` and return an :class:`AgentResult`.

        Args:
            task: The task text. Ignored when ``resume_from`` is given --
                a resumed run continues the transcript it already has.
            cancellation_event: Checked at step boundaries and around each
                tool call.
            resume_from: A checkpoint from a previously suspended run (see
                :attr:`AgentResult.checkpoint`). The transcript, trajectory
                and consumed budget are restored, and the context providers
                are **not** re-run: they prepared this conversation once
                already, and replaying them would inject a second copy of
                the same retrieved evidence.
        """
        generator = self.iter_run(
            task, cancellation_event=cancellation_event, resume_from=resume_from
        )
        while True:
            try:
                next(generator)
            except StopIteration as done:
                return done.value

    def iter_run(
        self,
        task: str,
        cancellation_event: Optional[threading.Event] = None,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> Iterator[RunEvent]:
        """Run the agent, yielding a :class:`RunEvent` as each thing happens.

        Returns the :class:`AgentResult` as the generator's value, so
        :meth:`run` is exactly ``drain this and take the result``.

        This exists because ``/api/stream`` needs to observe a run *while*
        it happens, and the only previous way to do that was to
        re-implement the loop inline in the HTTP layer. The two copies then
        drifted apart -- see BUGS.md #22. Emitting events is the smallest
        thing that makes one implementation serve both, and it keeps the
        difference between the endpoints where it belongs: in transport.

        Nodes are executed here rather than by the graph, because a node
        does not perform its own I/O any more -- it yields a description of
        the call it needs (see :mod:`.events`) and this driver makes it.
        :meth:`aiter_run` is the same loop with an awaiting driver.
        """

        ctx = self._prepare_context(task, resume_from)
        started = time.perf_counter()
        with bind_log_context(run_id=ctx.run_id, agent_name=self.agent_name), run_log_file(
            ctx.run_id
        ):
            self._log_started(task)
            try:
                state = self._initial_state(ctx, cancellation_event)
                yield RunEvent(RUN_STARTED, {"run_id": ctx.run_id, "task": task})

                traversal = self._graph.iter_steps(self._max_transitions())
                try:
                    _name, node = next(traversal)
                    while True:
                        state = yield from self._drive(node, state, ctx)
                        try:
                            _name, node = traversal.send(state)
                        except StopIteration:
                            break
                finally:
                    traversal.close()

                result = self._assemble(task, state, ctx)
                yield RunEvent(RUN_COMPLETED, self._completion_payload(result))
                self._log_completed(result, started)
                return result
            except Exception:
                log_event(
                    logger,
                    logging.ERROR,
                    "agent.run.failed",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    exc_info=True,
                )
                raise

    async def aiter_run(
        self,
        task: str,
        cancellation_event: Optional[threading.Event] = None,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> AsyncIterator[RunEvent]:
        """:meth:`iter_run` for a caller that owns an event loop.

        Identical traversal, identical nodes; the only difference is the
        driver. Tool dispatch runs in a worker thread, so a slow tool cannot
        stall every other request sharing the process (measured before this
        existed: a 300ms tool delayed a 30ms coroutine to 342ms), and a
        provider with real streaming produces :data:`TEXT` events as tokens
        arrive.

        The final :class:`AgentResult` cannot be returned -- an async
        generator may not return a value -- so it rides on the terminating
        :data:`RUN_COMPLETED` event's ``result`` key.
        """

        # Context providers retrieve (RAG) and summarize (session history);
        # _assemble writes durable memory, which can embed. Both are provider
        # I/O, and neither belongs on the event loop.
        ctx = await asyncio.to_thread(self._prepare_context, task, resume_from)
        started = time.perf_counter()
        with bind_log_context(run_id=ctx.run_id, agent_name=self.agent_name), run_log_file(
            ctx.run_id
        ):
            self._log_started(task)
            try:
                state = self._initial_state(ctx, cancellation_event)
                yield RunEvent(RUN_STARTED, {"run_id": ctx.run_id, "task": task})

                traversal = self._graph.iter_steps(self._max_transitions())
                try:
                    _name, node = next(traversal)
                    while True:
                        finished: List[Dict[str, Any]] = []
                        async for event in self._adrive(node, state, ctx, finished):
                            yield event
                        state = finished[0]
                        try:
                            _name, node = traversal.send(state)
                        except StopIteration:
                            break
                finally:
                    traversal.close()

                result = await asyncio.to_thread(self._assemble, task, state, ctx)
                payload = self._completion_payload(result)
                payload["result"] = result
                yield RunEvent(RUN_COMPLETED, payload)
                self._log_completed(result, started)
            except Exception:
                log_event(
                    logger,
                    logging.ERROR,
                    "agent.run.failed",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    exc_info=True,
                )
                raise

    # -- drivers ------------------------------------------------------------
    def _drive(
        self, node: Any, state: Dict[str, Any], ctx: ExecutionContext
    ) -> Iterator[RunEvent]:
        """Run one generator node on this thread, satisfying its effects.

        Yields the node's events straight through; performs anything else it
        yields and sends the result back in. Exceptions from an effect are
        thrown *into* the node rather than propagated, so a node's own
        ``try/except`` around its I/O keeps working exactly as it read when
        the call was inline -- that is where ``TransientLLMError`` becomes a
        graceful stop and ``SuspendRun`` becomes a checkpoint.
        """

        generator = node(state)
        to_send: Any = None
        to_throw: Optional[BaseException] = None
        while True:
            try:
                if to_throw is not None:
                    request = generator.throw(to_throw)
                    to_throw = None
                else:
                    request = generator.send(to_send)
            except StopIteration as done:
                return done.value if done.value is not None else state
            to_send = None
            if isinstance(request, RunEvent):
                yield request
                continue
            try:
                to_send = self._perform(request, ctx)
            except Exception as exc:  # noqa: BLE001 - handed to the node
                to_throw = exc

    async def _adrive(
        self,
        node: Any,
        state: Dict[str, Any],
        ctx: ExecutionContext,
        finished: List[Dict[str, Any]],
    ) -> AsyncIterator[RunEvent]:
        """:meth:`_drive` with an awaiting effect performer.

        The finished state is appended to ``finished`` rather than returned,
        because an async generator cannot return a value.
        """

        generator = node(state)
        to_send: Any = None
        to_throw: Optional[BaseException] = None
        while True:
            try:
                if to_throw is not None:
                    request = generator.throw(to_throw)
                    to_throw = None
                else:
                    request = generator.send(to_send)
            except StopIteration as done:
                finished.append(done.value if done.value is not None else state)
                return
            to_send = None
            if isinstance(request, RunEvent):
                yield request
                continue
            produced: List[Any] = []
            try:
                async for event in self._aperform(request, ctx, produced):
                    yield event
            except Exception as exc:  # noqa: BLE001 - handed to the node
                to_throw = exc
            else:
                to_send = produced[0] if produced else None

    def _perform(self, effect: Any, ctx: ExecutionContext) -> Any:
        """Satisfy one effect by blocking on it."""

        if isinstance(effect, ManageContext):
            return self._manage_context(ctx)
        if isinstance(effect, CallModel):
            return self.llm.chat(effect.messages, tools=effect.tools)
        if isinstance(effect, CallTool):
            return self._dispatcher.dispatch(ctx, effect.name, effect.arguments)
        raise TypeError(f"A node yielded something that is not an effect: {effect!r}")

    def _manage_context(
        self, ctx: ExecutionContext
    ) -> Tuple[List[Dict[str, Any]], int]:
        """The prompt for the next call, plus how many tokens compression saved.

        Both halves can call the model (summarizing the overflow window,
        compressing tool output), which is why this is reached through an
        effect rather than run inline in the node.
        """

        managed = self.short_term.manage(ctx.messages)
        if self.compressor is not None and self._near_budget(managed):
            managed, saved = self.compressor.compress_messages(
                managed, self._latest_user(ctx)
            )
            return managed, saved
        return managed, 0

    async def _aperform(
        self, effect: Any, ctx: ExecutionContext, produced: List[Any]
    ) -> AsyncIterator[RunEvent]:
        """Satisfy one effect from an event loop, appending the result.

        An async generator rather than a coroutine so a streamed model call
        can emit :data:`TEXT` events as tokens arrive instead of after.
        """

        if isinstance(effect, ManageContext):
            produced.append(await asyncio.to_thread(self._manage_context, ctx))
            return

        if isinstance(effect, CallTool):
            produced.append(
                await asyncio.to_thread(
                    self._dispatcher.dispatch, ctx, effect.name, effect.arguments
                )
            )
            return

        if not isinstance(effect, CallModel):
            raise TypeError(f"A node yielded something that is not an effect: {effect!r}")

        if not _streams(self.llm):
            # BaseLLM.astream's default just wraps chat(), so calling it here
            # would block the loop while pretending to stream. Be honest and
            # put the blocking call in a thread.
            produced.append(
                await asyncio.to_thread(self.llm.chat, effect.messages, effect.tools)
            )
            return

        content = ""
        tool_calls: List[ToolCall] = []
        async for chunk in self.llm.astream(effect.messages, tools=effect.tools):
            if chunk.get("type") == "text":
                content += chunk["data"]
                yield RunEvent(TEXT, {"step": effect.step_index, "token": chunk["data"]})
            elif chunk.get("type") == "tool_call":
                data = chunk["data"]
                tool_calls.append(
                    ToolCall(
                        id=data.get("id", ""),
                        name=data.get("name", ""),
                        arguments=data.get("arguments"),
                    )
                )
        produced.append(
            LLMResponse(
                content=content,
                tool_calls=tool_calls,
                # Streaming responses carry no usage block from any provider
                # this repo talks to. Estimating both halves keeps the token
                # budget working; `estimated` keeps it from being reported as
                # measured. See BUGS.md #22 symptom 3, where the streaming
                # path counted only the output text and a ~1000-token prompt
                # was recorded as 5.
                usage=Usage(
                    prompt_tokens=_prompt_tokens(effect.messages, effect.tools),
                    completion_tokens=estimate_tokens(content),
                    estimated=True,
                ),
            )
        )

    # -- run scaffolding ----------------------------------------------------
    def _prepare_context(
        self, task: str, resume_from: Optional[Dict[str, Any]]
    ) -> ExecutionContext:
        if resume_from is not None:
            return ExecutionContext.restore(resume_from)
        ctx = ExecutionContext(max_steps=self.max_steps, max_tokens=self.max_tokens)
        ctx.add_message("system", self.system_prompt)
        for provider in self.context_providers:
            for message in provider.prepare(task):
                role = str(message.get("role", "system"))
                content = str(message.get("content", ""))
                if content:
                    ctx.add_message(role, content)
        ctx.add_message("user", task)
        return ctx

    def _initial_state(
        self, ctx: ExecutionContext, cancellation_event: Optional[threading.Event]
    ) -> Dict[str, Any]:
        return {
            "ctx": ctx,
            "loop": self,
            "__cancellation_event__": cancellation_event,
        }

    def _assemble(
        self, task: str, state: Dict[str, Any], ctx: ExecutionContext
    ) -> AgentResult:
        answer: Optional[str] = state.get("__answer__")
        stop_reason: str = state.get("__stop_reason__", "finished")

        if answer is None:
            answer = self._force_finish(ctx)
            if stop_reason == "finished":
                stop_reason = "no_answer"

        suspended = stop_reason == SUSPENDED_STOP_REASON
        result = AgentResult(
            answer=answer,
            success=stop_reason in ("finished",),
            steps=len(ctx.steps),
            tokens=ctx.tokens_used,
            stop_reason=stop_reason,
            trajectory=ctx.trajectory(),
            checkpoint=ctx.checkpoint() if suspended else None,
            pending_job_ids=list(state.get("__pending_job_ids__", [])),
        )
        if not suspended:
            # A suspended run has not completed, so it has nothing to offer
            # durable memory yet -- writing now would record a half-finished
            # conclusion as a fact.
            self._record_memory_event(task, result, ctx)
        return result

    def _max_transitions(self) -> int:
        """The graph's safety net -- deliberately far above any real budget.

        ``StateGraph``'s own default is 200, which is a fine number for
        "this graph cannot terminate" and a bad one for this graph: a step
        costs two transitions (think, act) and three when a tool failure
        forces a reflection turn, so 200 fires at roughly 66 steps. The
        shipped ``leader.max_steps`` is **100**, so a long run hit the graph
        cap first and reported ``max_transitions (200)`` -- a number about
        the engine, not about the run -- instead of its real stop reason.
        See BUGS.md #23.

        Computed per run rather than at construction because ``max_steps``
        is writable (see ``ReActAgent``'s delegating properties).
        ``ExecutionContext.over_budget()`` remains the authoritative guard;
        this only catches a graph that cannot terminate at all.
        """

        return max(200, self.max_steps * 4 + 20)

    def _log_started(self, task: str) -> None:
        log_event(
            logger,
            logging.INFO,
            "agent.run.started",
            task_chars=len(task),
            max_steps=self.max_steps,
            max_tokens=self.max_tokens,
            tool_count=len(self.tools),
        )

    def _log_completed(self, result: AgentResult, started: float) -> None:
        log_event(
            logger,
            logging.INFO,
            "agent.run.completed",
            success=result.success,
            stop_reason=result.stop_reason,
            steps=result.steps,
            tokens=result.tokens,
            answer_chars=len(result.answer),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    @staticmethod
    def _completion_payload(result: AgentResult) -> Dict[str, Any]:
        return {
            "answer": result.answer,
            "success": result.success,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "tokens": result.tokens,
            "pending_job_ids": list(result.pending_job_ids),
        }

    def _record_memory_event(
        self,
        task: str,
        result: AgentResult,
        ctx: ExecutionContext,
    ) -> None:
        """Offer a completed run to policy-controlled memory without risking the run."""

        if self.memory_manager is None:
            return
        try:
            stored = self.memory_manager.on_run_completed(
                RunCompletedEvent(
                    run_id=ctx.run_id,
                    task=task,
                    answer=result.answer,
                    success=result.success,
                    stop_reason=result.stop_reason,
                    messages=list(ctx.messages),
                    trajectory=result.trajectory,
                    namespace=self.memory_namespace,
                    subject_id=self.memory_subject_id,
                )
            )
            log_event(
                logger,
                logging.INFO,
                "memory.run.processed",
                candidate_stored_count=len(stored),
            )
        except Exception:
            # Durable memory is best-effort here. Critical business writes should
            # use an application outbox rather than making answer delivery depend
            # on an embedding service or vector index.
            log_event(
                logger,
                logging.WARNING,
                "memory.run.failed",
                exc_info=True,
            )

    # -- graph nodes --------------------------------------------------------
    # Both nodes are generators. Everything they do is pure except the one
    # call in the middle, which they ``yield`` as an effect and get the
    # result of back (see :mod:`.events`). That is what lets a blocking
    # driver and an awaiting driver share one node body: a ``def`` cannot
    # ``await``, but it can ``yield``. They also ``yield`` RunEvents, which
    # the driver forwards without sending anything back.
    def _think_node(self, state: Dict[str, Any]) -> Iterator[Any]:
        """Graph node: one LLM call with memory management and compression."""
        ctx: ExecutionContext = state["ctx"]

        cancellation_event = state.get("__cancellation_event__")
        if cancellation_event is not None and cancellation_event.is_set():
            state["__next__"] = "finish"
            state["__answer__"] = None
            state["__stop_reason__"] = "cancelled"
            return state

        if ctx.over_budget():
            state["__next__"] = "finish"
            state["__answer__"] = None
            state["__stop_reason__"] = _budget_stop_reason(ctx)
            return state

        if self.loop_detection:
            loop_reason = _detect_loop(ctx.steps, self.loop_same_call_limit)
            if loop_reason:
                state["__next__"] = "finish"
                state["__answer__"] = None
                state["__stop_reason__"] = f"loop_detected: {loop_reason}"
                return state

        reflect = ctx.state.pop(REFLECT_AFTER_FAILURE_STATE_KEY, False)

        step = ctx.new_step()

        managed, saved = yield ManageContext()
        if saved:
            ctx.state["tokens_saved_by_compression"] = (
                ctx.state.get("tokens_saved_by_compression", 0) + saved
            )

        offered = None if reflect else self._offered_schemas(ctx)

        # The token budget is checked again here, against the prompt this
        # call is about to send. ``over_budget()`` above only sees tokens
        # already *spent*, so a run one token under its ceiling could still
        # issue a call with a 40k-token prompt: max_tokens=1 with a
        # 101-token call finished successfully, reporting tokens=101.
        # Counting the prompt first turns an unbounded overshoot into one
        # bounded by the model's max output. It cannot be made exact --
        # nothing knows the completion size before the call -- so this stays
        # a ceiling with a known overshoot, not a hard cap. See BUGS.md #21.
        projected = ctx.tokens_used + _prompt_tokens(managed, offered)
        if projected >= self.max_tokens:
            ctx.steps.pop()
            state["__next__"] = "finish"
            state["__answer__"] = None
            state["__stop_reason__"] = (
                f"budget: max_tokens ({self.max_tokens}) would be exceeded by "
                f"this call (projected {projected})"
            )
            return state

        llm_started = time.perf_counter()
        log_event(
            logger,
            logging.DEBUG,
            "llm.call.started",
            step=step.index,
            message_count=len(managed),
            tool_count=0 if offered is None else len(offered),
            registry_size=len(self.tools),
            forced_reflection=reflect,
        )
        yield RunEvent(THINK_STARTED, {"step": step.index})
        try:
            response = yield CallModel(managed, offered, step.index)
        except TransientLLMError as exc:
            # The provider was unreachable/overloaded and
            # agent.retry.call_with_retry already spent its attempt budget.
            # Ending the run here rather than re-raising keeps the work done
            # so far -- the trajectory, the tool observations already
            # collected, the token accounting -- which an exception
            # propagating out of run() would discard entirely. The caller
            # still sees failure via stop_reason; it just also gets the
            # partial result. A PermanentLLMError deliberately keeps
            # propagating: a request the provider rejects outright is a bug
            # to fix (see BUGS.md #1), not a condition to degrade around.
            log_event(
                logger,
                logging.ERROR,
                "llm.call.unavailable",
                step=step.index,
                elapsed_ms=round((time.perf_counter() - llm_started) * 1000, 2),
                exc_info=True,
            )
            step.error = str(exc)
            yield RunEvent(
                ERROR,
                {
                    "step": step.index,
                    "type": "llm_unavailable",
                    "message": str(exc),
                },
            )
            state["__next__"] = "finish"
            state["__answer__"] = None
            state["__stop_reason__"] = "llm_unavailable"
            return state
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "llm.call.failed",
                step=step.index,
                elapsed_ms=round((time.perf_counter() - llm_started) * 1000, 2),
                exc_info=True,
            )
            raise
        log_event(
            logger,
            logging.INFO,
            "llm.call.completed",
            step=step.index,
            wants_tool=response.wants_tool,
            tool_call_count=len(response.tool_calls),
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            estimated_usage=response.usage.estimated,
            elapsed_ms=round((time.perf_counter() - llm_started) * 1000, 2),
        )
        ctx.add_tokens(response.usage.total_tokens)
        yield RunEvent(
            THINK_COMPLETED,
            {
                "step": step.index,
                "thought": response.content or "",
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                    for tc in response.tool_calls
                ],
                "tokens": response.usage.total_tokens,
                "estimated_usage": response.usage.estimated,
            },
        )

        if reflect:
            # No tools were offered, so this response can only be text --
            # never a final answer and never a tool call, regardless of what
            # response.wants_tool/content look like. Record it as a plain
            # reasoning step and go straight back to a normal think step
            # (tools available again) rather than finishing the run.
            content = (response.content or "").strip() or "(no reasoning provided)"
            step.thought = content
            step.observation = "reflection"
            ctx.add_message("assistant", content)
            yield RunEvent(REFLECTION, {"step": step.index, "text": content})
            state["__next__"] = "think"
            return state

        if response.wants_tool:
            step.thought = response.content or "(calling tool)"
            state["__pending__"] = {
                "tool_calls": response.tool_calls,
                "step": step,
            }
            state["__next__"] = "tools"
        else:
            answer = (response.content or "").strip()
            step.thought = answer
            step.observation = "final_answer"
            ctx.add_message("assistant", answer)
            state["__answer__"] = answer
            state["__next__"] = "finish"

        return state

    def _act_node(self, state: Dict[str, Any]) -> Iterator[Any]:
        """Graph node: dispatch tool calls and record observations."""
        ctx: ExecutionContext = state["ctx"]

        pending = state.pop("__pending__", None)
        if pending is None:
            state["__next__"] = "think"
            return state

        tool_calls: List[ToolCall] = pending["tool_calls"]
        step: Step = pending["step"]

        ctx.add_message(
            "assistant",
            step.thought,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tool_calls
            ],
        )

        cancellation_event = state.get("__cancellation_event__")
        failed: List[str] = []
        for tc in tool_calls:
            # Checked here as well as at the top of _think_node: a run
            # cancelled while a long tool call is in flight would otherwise
            # go on to dispatch every *remaining* call in the same step
            # before the loop next looked at the event.
            if cancellation_event is not None and cancellation_event.is_set():
                state["__next__"] = "finish"
                state["__answer__"] = None
                state["__stop_reason__"] = "cancelled"
                return state
            yield RunEvent(
                TOOL_STARTED,
                {
                    "step": step.index,
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                },
            )
            try:
                observation = yield CallTool(tc.name, tc.arguments)
            except SuspendRun as signal:
                # Not a failure: the run is waiting on work that is
                # deliberately not on this call stack. Record what has been
                # observed so far and hand the caller a resumable
                # checkpoint -- see agent/jobs/.
                if step.action is None:
                    step.action = {"name": tc.name, "arguments": tc.arguments}
                log_event(
                    logger,
                    logging.INFO,
                    "agent.run.suspended",
                    pending_job_ids=signal.job_ids,
                    step=step.index,
                )
                state["__pending_job_ids__"] = signal.job_ids
                state["__answer__"] = None
                state["__stop_reason__"] = SUSPENDED_STOP_REASON
                state["__next__"] = "finish"
                yield RunEvent(
                    SUSPENDED,
                    {"step": step.index, "pending_job_ids": list(signal.job_ids)},
                )
                return state
            except FatalToolError as exc:
                message = str(exc)
                step.record_tool_call(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.arguments,
                    observation=None,
                    ok=False,
                )
                step.error = message
                step.observation = None
                state["__answer__"] = f"Fatal tool error: {message}"
                state["__stop_reason__"] = "fatal_tool_error"
                state["__next__"] = "finish"
                yield RunEvent(
                    ERROR,
                    {
                        "step": step.index,
                        "type": "fatal_tool_error",
                        "message": message,
                    },
                )
                return state
            if self.output_guard is not None:
                scan = self.output_guard.scan(observation)
                if scan.suspicious:
                    observation = scan.sanitized
                    ctx.state["injection_flags"] = (
                        ctx.state.get("injection_flags", 0) + len(scan.matches)
                    )
            ok = not is_failure_observation(observation)
            if not ok:
                failed.append(tc.name)
            # Recorded per call, not per step: a turn that issues several
            # calls used to collapse into the first one's name plus every
            # observation concatenated. See Step's docstring and BUGS.md #16.
            step.record_tool_call(
                id=tc.id,
                name=tc.name,
                arguments=tc.arguments,
                observation=observation,
                ok=ok,
            )
            ctx.add_message("tool", observation, tool_call_id=tc.id, name=tc.name)
            yield RunEvent(
                TOOL_COMPLETED,
                {
                    "step": step.index,
                    "id": tc.id,
                    "name": tc.name,
                    "observation": observation,
                    "ok": ok,
                },
            )

        if failed:
            # Force the *next* think call to run with no tools offered (see
            # REFLECT_AFTER_FAILURE_STATE_KEY) instead of letting the model
            # go straight from a failed call to another tool call.
            ctx.add_message("user", FORCED_REFLECTION_PROMPT)
            ctx.state[REFLECT_AFTER_FAILURE_STATE_KEY] = True
            log_event(
                logger,
                logging.INFO,
                "reflection.forced",
                tool_name=failed[0],
                failed_tools=failed,
            )

        state["__next__"] = "think"
        return state

    # -- helpers ------------------------------------------------------------
    def _offered_schemas(self, ctx: ExecutionContext) -> List[Dict[str, Any]]:
        """The tool schemas this think step gets to see."""

        if self.tool_selector is None:
            return self.tools.schemas()
        selection = self.tool_selector.select(
            self.tools, _selection_query(ctx), used=_tools_used(ctx)
        )
        return filtered_schemas(self.tools, selection)

    def _near_budget(self, messages: List[Dict[str, Any]]) -> bool:
        prompt_tokens = sum(
            estimate_tokens(str(m.get("content") or "")) for m in messages
        )
        return prompt_tokens >= self.compress_at_fraction * self.max_tokens

    @staticmethod
    def _latest_user(ctx: ExecutionContext) -> str:
        for msg in reversed(ctx.messages):
            if msg.get("role") == "user":
                return str(msg.get("content") or "")
        return ""

    def _force_finish(self, ctx: ExecutionContext) -> str:
        for step in reversed(ctx.steps):
            if step.observation and step.observation not in ("final_answer", "reflection"):
                return f"(stopped) Last observation: {step.observation}"
        return "(stopped) No answer was produced within the configured budget."


# -- module-level helpers --------------------------------------------------
#: How many recent steps feed the tool-selection query. Small on purpose:
#: the whole transcript would drown the actual task in tool output, and the
#: task itself is the strongest signal for which tools are relevant.
_SELECTION_TRAJECTORY_WINDOW = 3


def _streams(llm: BaseLLM) -> bool:
    """Whether this provider actually streams, or just looks like it does.

    ``BaseLLM.astream`` has a default implementation that calls ``chat()``
    and yields the finished reply in one piece. That is the right fallback
    for a caller that only wants one interface, but the async driver has to
    tell the two apart: driving the default would block the event loop for
    the whole call while emitting a single "token" at the end -- the worst
    of both. Providers that genuinely stream override the method.
    """

    return type(llm).astream is not BaseLLM.astream


def _prompt_tokens(
    messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]]
) -> int:
    """Estimated input size of one call: the transcript plus the tool schemas.

    The schemas are not free -- a 56-tool registry is thousands of tokens of
    prompt on every step -- so a budget check that ignores them
    systematically under-counts exactly the runs most at risk of blowing it.
    """

    total = sum(estimate_tokens(str(m.get("content") or "")) for m in messages)
    if tools:
        total += sum(estimate_tokens(json.dumps(schema)) for schema in tools)
    return total


def _selection_query(ctx: ExecutionContext) -> str:
    """Text the tool selector matches against: the task plus recent progress.

    The task alone is not enough for a chain whose later steps need a tool
    the original question never mentions (fetch a page, *then* summarize
    it), so the last few thoughts and observations are appended.
    """

    parts: List[str] = []
    for message in ctx.messages:
        if message.get("role") == "user":
            parts.append(str(message.get("content") or ""))
            break
    for step in ctx.steps[-_SELECTION_TRAJECTORY_WINDOW:]:
        if step.thought:
            parts.append(step.thought)
        if step.observation and step.observation not in ("final_answer", "reflection"):
            parts.append(step.observation[:500])
    return "\n".join(part for part in parts if part)


def _tools_used(ctx: ExecutionContext) -> List[str]:
    """Tool names already called in this run; never hidden from later steps.

    Reads every recorded call, not just the step's flattened ``action``:
    the second tool of a parallel turn is exactly the kind the selector
    would otherwise be free to hide from the next step.
    """

    names: List[str] = []
    for step in ctx.steps:
        for call in step.tool_calls:
            name = call.get("name")
            if name and name not in names:
                names.append(str(name))
    return names


def _detect_loop(steps: List[Step], same_call_limit: int = 3) -> Optional[str]:
    """Return a reason if the trajectory is looping, else ``None``.

    Distinct from the budget guardrail: the step/token budget is a hard ceiling
    that stops the run no matter what, while this inspects the *content* of the
    trajectory to decide whether the agent is re-issuing work instead of making
    progress. Two deterministic signals, both read from the recorded steps (no
    extra LLM calls, no dependencies):

    * **repeated call** — the most recent ``(tool, arguments)`` has been issued
      ``same_call_limit`` times in total (this also catches N-cycles);
    * **oscillation** — the last four calls alternate ``A -> B -> A -> B``.

    Arguments are canonicalized (JSON, sorted keys, lowercased) so equivalent
    spellings of a call count as the same call.

    A known blind spot: a loop that *varies* its arguments each time (e.g.
    ``page=1``, ``page=2``, ...) is not caught here; that needs an
    observation-novelty signal, not a call-identity one.
    """

    calls: List[Tuple[str, str]] = []
    for s in steps:
        action = s.action
        if not action:
            continue
        name = action.get("name")
        args = action.get("arguments") or {}
        try:
            canon = json.dumps(args, sort_keys=True).lower()
        except (TypeError, ValueError):
            canon = str(args).lower()
        calls.append((name, canon))

    if not calls:
        return None

    name, canon = calls[-1]
    if sum(1 for c in calls if c == (name, canon)) >= same_call_limit:
        return f"repeated call '{name}' x{same_call_limit}"

    if len(calls) >= 4 and calls[-1] == calls[-3] and calls[-2] == calls[-4]:
        return f"oscillating '{calls[-2][0]}' <-> '{calls[-1][0]}'"

    return None


def _route_by_next(state: Dict[str, Any]) -> str:
    return state.get("__next__", "finish")


def _budget_stop_reason(ctx: ExecutionContext) -> str:
    reason = ctx.budget_reason()
    return f"budget: {reason}" if reason else "budget"
