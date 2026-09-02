"""Hierarchical (main-agent + specialist-subagent) tool routing.

Motivation: the classic "too many tools" complaint is that a flat registry
of hundreds of tools bloats every single LLM call with every tool's full
schema, which is what actually degrades selection accuracy in the wild
(most public write-ups; see ``tool_scaling_kit.py``'s docstring). This
module implements the standard mitigation: namespace tools behind a small
number of specialist subagents.

    1. The main agent only ever sees a handful of *base* tools plus one
       "delegate" tool per specialist -- and each delegate tool's
       description is a short one-liner ("handles arithmetic and
       statistics"), not the specialist's full tool list.
    2. Calling a delegate tool spins up a **fresh** :class:`ReActAgent`
       whose registry is *only* that specialist's tools, each with its
       full, detailed description (a small registry can afford verbose
       descriptions without bloating the top-level context).
    3. If a specialist can't complete the delegated sub-task with the
       tools it has, its system prompt instructs it to say so explicitly
       (``TASK_FAILED: <reason>``) rather than guess.

This directly attacks the axis ``tool_scaling_test.py`` found *no* effect
on up to 100 flat tools (single-tool selection stayed ~100%) and the axis
``tool_scaling_verbose_test.py`` found little effect on (~6.5x description
bloat). Whether it helps the axis that *did* show a real, reproducible
effect -- long chains (5+ steps) causing the model to silently skip
self-judged-redundant steps, see ``tool_scaling_long_chain_test.py`` -- is
the open question ``hierarchical_agent_test.py`` measures by rerunning the
exact same 21 five-step tasks through this hierarchy instead of the flat
100-tool registry.

Five specialists, two of the kit's ten categories each -- broad enough to
mirror how a real deployment would carve up "calculation", "text",
"date/calendar", "conversion", and "data/encoding" capabilities:

    calculate_agent -- math_*, stat_*        (20 tools)
    text_agent      -- text_*, format_*      (20 tools)
    datetime_agent  -- date_*, calendar_*    (20 tools)
    convert_agent   -- convert_*, measure_*  (20 tools)
    data_agent      -- data_*, encode_*      (20 tools)
"""

from __future__ import annotations

import os
import sys
from typing import Any, Callable, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> local imports

from agent import ReActAgent, ToolRegistry, tool  # noqa: E402
from agent.llm import BaseLLM  # noqa: E402
from agent.tools import BaseTool  # noqa: E402

from tool_scaling_kit import ALL_TOOLS  # noqa: E402

# ---------------------------------------------------------------------------
# Base tools -- what the *main* agent sees directly, unrelated to the kit.
# Deliberately trivial stand-ins matching the "fetch / get_date" examples;
# none of the 21 test tasks need them, they just occupy the "main agent has
# a couple of basic tools of its own" slot the design calls for.
# ---------------------------------------------------------------------------


@tool
def fetch(query: str) -> str:
    """Look up a fact from a small canned knowledge base (stands in for a
    real web-fetch tool).
    Args:
        query: What to look up.
    """
    return f"No canned result for '{query}'."


@tool
def get_date() -> str:
    """Return today's date in ISO-8601 format (UTC)."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


BASE_TOOLS: List[BaseTool] = [fetch, get_date]

# ---------------------------------------------------------------------------
# Specialist groups
# ---------------------------------------------------------------------------

GROUP_DEFS: Dict[str, Dict[str, Any]] = {
    "calculate_agent": {
        "description": (
            "Handles arithmetic and statistics: add, subtract, multiply, "
            "divide, rounding, sum, average, min, max, and similar numeric "
            "operations."
        ),
        "prefixes": ("math_", "stat_"),
    },
    "text_agent": {
        "description": (
            "Handles text processing: case conversion, reversing, "
            "counting characters/words/vowels, replacing substrings, "
            "palindrome checks, and text formatting such as snake_case."
        ),
        "prefixes": ("text_", "format_"),
    },
    "datetime_agent": {
        "description": (
            "Handles dates and calendars: adding/subtracting days, "
            "day-of-week, days-between, weekend checks, ISO week numbers, "
            "and leap-year checks."
        ),
        "prefixes": ("date_", "calendar_"),
    },
    "convert_agent": {
        "description": (
            "Handles unit conversion: distance (kilometers/miles/nautical "
            "miles), weight (kilograms/pounds), and temperature "
            "(Celsius/Fahrenheit)."
        ),
        "prefixes": ("convert_", "measure_"),
    },
    "data_agent": {
        "description": (
            "Handles structured data and encoding: JSON validation, list "
            "dedupe/sort, and Base64/hex encode-decode."
        ),
        "prefixes": ("data_", "encode_"),
    },
}


def _group_for_tool(name: str) -> Optional[str]:
    for group_name, spec in GROUP_DEFS.items():
        if name.startswith(spec["prefixes"]):
            return group_name
    return None


def _build_subagent_tools() -> Dict[str, List[BaseTool]]:
    """Partition ALL_TOOLS into the five specialist groups."""

    grouped: Dict[str, List[BaseTool]] = {name: [] for name in GROUP_DEFS}
    unassigned: List[str] = []
    for t in ALL_TOOLS:
        group_name = _group_for_tool(t.name)
        if group_name is None:
            unassigned.append(t.name)
        else:
            grouped[group_name].append(t)
    if unassigned:
        raise RuntimeError(f"Tools with no specialist group: {unassigned}")
    return grouped


SUBAGENT_TOOLS: Dict[str, List[BaseTool]] = _build_subagent_tools()

SUBAGENT_SYSTEM_PROMPT = (
    "You are {group_name}, a specialist agent. Reason step by step and use "
    "the provided tools to complete the delegated sub-task precisely -- "
    "call every tool call the task actually requires, even one whose "
    "result looks like it wouldn't change (e.g. rounding a number that is "
    "already at the target precision); do not skip a step just because you "
    "are confident you already know the answer. If, and only if, none of "
    "your available tools can complete the sub-task, respond with exactly: "
    "TASK_FAILED: <short reason>. Otherwise respond with a final answer "
    "and do not call any more tools."
)

MAIN_SYSTEM_PROMPT = (
    "You are a coordinator agent. You do not have direct tools for "
    "calculation, text processing, dates, unit conversion, or data "
    "encoding -- for those you must delegate to the matching specialist "
    "tool (delegate_calculate_agent, delegate_text_agent, "
    "delegate_datetime_agent, delegate_convert_agent, delegate_data_agent). "
    "Break the user's request into an ordered sequence of sub-tasks, one "
    "per specialist, and delegate each in turn -- pass the specialist the "
    "concrete numbers/text it needs (including any result a previous "
    "delegate call returned), since it cannot see the rest of the "
    "conversation. Chain as many delegate calls as the request needs. When "
    "you have the final result, respond with a final answer and do not "
    "call any more tools."
)


class DelegateTool(BaseTool):
    """A main-agent tool that forwards a sub-task to a specialist subagent.

    The main agent only ever sees ``name``/``description`` (one short
    line); the specialist's full tool list and detailed descriptions live
    entirely inside the nested :class:`ReActAgent` this spins up per call.
    """

    def __init__(
        self,
        group_name: str,
        description: str,
        tools: List[BaseTool],
        llm_factory: Callable[[], BaseLLM],
        call_log: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = 8,
        subagent_system_prompt: Optional[str] = None,
    ) -> None:
        self.group_name = group_name
        self.name = f"delegate_{group_name}"
        self.description = description
        self._tools = tools
        self._llm_factory = llm_factory
        self._call_log = call_log if call_log is not None else []
        self._max_steps = max_steps
        # Overridable so experiments can vary the specialist's instructions --
        # the diagnostic in INTERVENTION_LADDER.md showed every skipped
        # step is dropped *inside* a specialist that was correctly delegated
        # to, so this, not MAIN_SYSTEM_PROMPT, is the layer that matters.
        # Must contain a {group_name} field.
        self._subagent_system_prompt = (
            subagent_system_prompt or SUBAGENT_SYSTEM_PROMPT
        )

    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The exact sub-task to perform, in natural "
                        "language, including any concrete numbers or text "
                        "values it needs."
                    ),
                }
            },
            "required": ["task"],
        }

    def run(self, task: str) -> str:
        registry = ToolRegistry(list(self._tools))
        llm = self._llm_factory()
        sub_agent = ReActAgent(
            llm=llm,
            tools=registry,
            system_prompt=self._subagent_system_prompt.format(group_name=self.group_name),
            max_steps=self._max_steps,
            agent_name=self.group_name,
        )
        outcome = sub_agent.run(task)
        called = [
            (step.get("action") or {}).get("name")
            for step in outcome.trajectory if step.get("action")
        ]
        self._call_log.append({
            "group": self.group_name,
            "task": task,
            "called_tools": called,
            "answer": outcome.answer,
            "stop_reason": outcome.stop_reason,
        })
        return outcome.answer


def build_main_registry(
    llm_factory: Callable[[], BaseLLM],
    call_log: Optional[List[Dict[str, Any]]] = None,
    max_steps: int = 8,
    subagent_system_prompt: Optional[str] = None,
) -> ToolRegistry:
    """Build the main agent's registry: base tools + one delegate per group.

    ``subagent_system_prompt`` overrides what every specialist is told (it
    must contain a ``{group_name}`` field). Defaults to
    :data:`SUBAGENT_SYSTEM_PROMPT`.
    """

    if call_log is None:
        call_log = []
    registry = ToolRegistry(list(BASE_TOOLS))
    for group_name, spec in GROUP_DEFS.items():
        registry.register(DelegateTool(
            group_name=group_name,
            description=spec["description"],
            tools=SUBAGENT_TOOLS[group_name],
            llm_factory=llm_factory,
            call_log=call_log,
            max_steps=max_steps,
            subagent_system_prompt=subagent_system_prompt,
        ))
    return registry


def main() -> None:
    """Print a schema-size comparison: flat 100-tool registry vs. what the
    main agent sees in the hierarchical setup (base tools + 5 one-liners)."""

    import json

    flat_registry = ToolRegistry(list(ALL_TOOLS))
    flat_schema = json.dumps(flat_registry.schemas())

    call_log: List[Dict[str, Any]] = []
    main_registry = build_main_registry(llm_factory=lambda: None, call_log=call_log)
    main_schema = json.dumps(main_registry.schemas())

    print(f"Flat registry:         {len(flat_registry)} tools, {len(flat_schema)} schema chars")
    print(f"Main agent (hierarchy): {len(main_registry)} tools, {len(main_schema)} schema chars")
    print(f"Schema-size reduction:  {len(flat_schema) / len(main_schema):.1f}x smaller")
    print()
    for group_name, tools in SUBAGENT_TOOLS.items():
        print(f"  {group_name:<18} {len(tools)} tools -- {GROUP_DEFS[group_name]['description']}")


if __name__ == "__main__":
    main()
