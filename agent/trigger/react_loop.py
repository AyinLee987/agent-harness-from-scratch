"""ReAct loop — the think → act → observe cycle.

Extracted from :class:`~agent.ReActAgent`.  The :class:`ReActLoop` owns the
graph construction, the think/act node implementations, and the main ``run()``
entry point.  It is wired together with a :class:`StateGraph` from
:mod:`.graph` and a :class:`ToolDispatcher` from :mod:`.dispatch`.

Guardrails included: a hard step/token budget, a finish check, and
retry-once-then-fail handling of malformed tool calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..compression import ContextCompressor
from ..errors import FatalToolError
from ..llm import BaseLLM, LLMResponse, ToolCall, estimate_tokens
from ..safety import ToolOutputGuard
from ..state.context import ExecutionContext, Step
from ..state.memory import LongTermMemory, ShortTermMemory
from ..tools import ToolRegistry
from .dispatch import ToolDispatcher
from .graph import StateGraph

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful ReAct agent. Reason step by step. Use the provided tools "
    "when they help answer the user's request. If a memory_search tool is available "
    "and the user asks about something you might have stored from past conversations "
    "or domain knowledge, call it proactively. When you have enough information, "
    "respond with a final answer and do not call any more tools."
)


@dataclass
class AgentResult:
    """The outcome of an agent run."""

    answer: str
    success: bool
    steps: int
    tokens: int
    stop_reason: str
    trajectory: List[Dict[str, Any]]


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
        self._dispatcher = ToolDispatcher(tools, max_retries=max_tool_retries)
        self._graph = self._build_graph()

    # -- graph construction -------------------------------------------------
    def _build_graph(self) -> StateGraph:
        """Build the ReAct state graph.

        ::

            [think] ──(tool_calls)──▶ [act] ──(fixed)──▶ [think]
               │
               └──(answer)──▶ __end__
        """
        g = StateGraph()
        g.add_node("think", self._think_node)
        g.add_node("act", self._act_node)
        g.set_entry_point("think")
        g.add_conditional_edges(
            "think",
            _route_by_next,
            {"tools": "act", "finish": "__end__"},
        )
        g.add_conditional_edges(
            "act",
            _route_by_next,
            {"think": "think", "finish": "__end__"},
        )
        return g

    # -- public API ---------------------------------------------------------
    def run(self, task: str) -> AgentResult:
        """Run the agent to completion on ``task`` and return an :class:`AgentResult`."""
        ctx = ExecutionContext(max_steps=self.max_steps, max_tokens=self.max_tokens)
        ctx.add_message("system", self.system_prompt)
        ctx.add_message("user", task)

        state: Dict[str, Any] = {"ctx": ctx, "loop": self}
        executor = self._graph.compile()
        state = executor(state)

        answer: Optional[str] = state.get("__answer__")
        stop_reason: str = state.get("__stop_reason__", "finished")

        if answer is None:
            answer = self._force_finish(ctx)
            if stop_reason == "finished":
                stop_reason = "no_answer"

        if self.long_term is not None:
            self.long_term.add(f"Task: {task}\nAnswer: {answer}", {"type": "task"})

        return AgentResult(
            answer=answer,
            success=stop_reason in ("finished",),
            steps=len(ctx.steps),
            tokens=ctx.tokens_used,
            stop_reason=stop_reason,
            trajectory=ctx.trajectory(),
        )

    # -- graph nodes --------------------------------------------------------
    def _think_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Graph node: one LLM call with memory management and compression."""
        ctx: ExecutionContext = state["ctx"]

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

        step = ctx.new_step()

        managed = self.short_term.manage(ctx.messages)

        if self.compressor is not None and self._near_budget(managed):
            query = self._latest_user(ctx)
            managed, saved = self.compressor.compress_messages(managed, query)
            ctx.state["tokens_saved_by_compression"] = (
                ctx.state.get("tokens_saved_by_compression", 0) + saved
            )

        response = self.llm.chat(managed, tools=self.tools.schemas())
        ctx.add_tokens(response.usage.total_tokens)

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

    def _act_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
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

        observations: List[str] = []
        for tc in tool_calls:
            try:
                observation = self._dispatcher.dispatch(ctx, tc.name, tc.arguments)
            except FatalToolError as exc:
                message = str(exc)
                step.action = {"name": tc.name, "arguments": tc.arguments}
                step.error = message
                step.observation = None
                state["__answer__"] = f"Fatal tool error: {message}"
                state["__stop_reason__"] = "fatal_tool_error"
                state["__next__"] = "finish"
                return state
            if self.output_guard is not None:
                scan = self.output_guard.scan(observation)
                if scan.suspicious:
                    observation = scan.sanitized
                    ctx.state["injection_flags"] = (
                        ctx.state.get("injection_flags", 0) + len(scan.matches)
                    )
            observations.append(observation)
            ctx.add_message("tool", observation, tool_call_id=tc.id, name=tc.name)

        step.action = {
            "name": tool_calls[0].name,
            "arguments": tool_calls[0].arguments,
        }
        step.observation = "\n".join(observations)

        state["__next__"] = "think"
        return state

    # -- helpers ------------------------------------------------------------
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
            if step.observation and step.observation != "final_answer":
                return f"(stopped) Last observation: {step.observation}"
        return "(stopped) No answer was produced within the configured budget."


# -- module-level helpers --------------------------------------------------
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
