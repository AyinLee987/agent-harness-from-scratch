"""Tests for the hierarchical (main-agent + specialist-subagent) design.

Same split as the other tool_scaling test files: deterministic structural
checks always run; the live-LLM accuracy check is skip-gated on an API key
(routing decisions and cross-specialist orchestration are model decisions
MockLLM can't exercise meaningfully).
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = os.path.join(_REPO_ROOT, "examples")
sys.path.insert(0, _EXAMPLES)

from agent import ReActAgent  # noqa: E402

from hierarchical_agent_kit import (  # noqa: E402
    BASE_TOOLS,
    GROUP_DEFS,
    MAIN_SYSTEM_PROMPT,
    SUBAGENT_TOOLS,
    DelegateTool,
    build_main_registry,
)
from tool_scaling_kit import ALL_TOOLS  # noqa: E402

MULTI_TASKS_PATH = os.path.join(_EXAMPLES, "tool_scaling_multi_tasks.json")
LONG_CHAIN_TASKS_PATH = os.path.join(_EXAMPLES, "tool_scaling_long_chain_tasks.json")

HAS_LIVE_LLM = bool(
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("BAILIAN_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)


def _load_all_tasks():
    with open(MULTI_TASKS_PATH, "r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    with open(LONG_CHAIN_TASKS_PATH, "r", encoding="utf-8") as fh:
        tasks += json.load(fh)
    return tasks


# ---------------------------------------------------------------------------
# Structural checks (fast, free, always run)
# ---------------------------------------------------------------------------


def test_every_kit_tool_belongs_to_exactly_one_specialist_group():
    """SUBAGENT_TOOLS must partition ALL_TOOLS with no gaps or overlaps."""
    seen = set()
    for group_name, tools in SUBAGENT_TOOLS.items():
        for t in tools:
            assert t.name not in seen, f"{t.name} assigned to more than one group"
            seen.add(t.name)
    assert seen == {t.name for t in ALL_TOOLS}


def test_every_specialist_group_is_non_empty_and_covers_its_prefixes():
    for group_name, spec in GROUP_DEFS.items():
        tools = SUBAGENT_TOOLS[group_name]
        assert tools, f"{group_name} has no tools"
        for t in tools:
            assert t.name.startswith(spec["prefixes"]), (
                f"{t.name} in {group_name} doesn't match its declared prefixes {spec['prefixes']}"
            )


def test_group_descriptions_are_short_summaries_not_full_tool_lists():
    """The whole point is the main agent sees a one-liner, not every tool
    name/description -- so a group description should stay far shorter
    than the sum of its tools' own descriptions."""
    for group_name, spec in GROUP_DEFS.items():
        tools_desc_len = sum(len(t.description) for t in SUBAGENT_TOOLS[group_name])
        assert len(spec["description"]) < tools_desc_len / 4, (
            f"{group_name}'s description ({len(spec['description'])} chars) isn't "
            f"meaningfully shorter than its tools' combined descriptions."
        )


def test_delegate_tool_schema_takes_a_single_task_string():
    dt = DelegateTool(
        group_name="calculate_agent",
        description=GROUP_DEFS["calculate_agent"]["description"],
        tools=SUBAGENT_TOOLS["calculate_agent"],
        llm_factory=lambda: None,
    )
    schema = dt.parameters_schema()
    assert schema["required"] == ["task"]
    assert schema["properties"]["task"]["type"] == "string"


def test_main_registry_hides_the_full_kit_behind_delegate_tools():
    """The main agent's own registry must be small -- base tools plus one
    delegate tool per group -- never the 100 kit tools directly."""
    registry = build_main_registry(llm_factory=lambda: None)
    assert len(registry) == len(BASE_TOOLS) + len(GROUP_DEFS)
    names = set(registry.names())
    for group_name in GROUP_DEFS:
        assert f"delegate_{group_name}" in names
    for t in ALL_TOOLS:
        assert t.name not in names, f"{t.name} leaked into the main registry"


def test_main_registry_schema_is_much_smaller_than_the_flat_kit():
    from agent import ToolRegistry
    flat = ToolRegistry(list(ALL_TOOLS))
    hierarchical = build_main_registry(llm_factory=lambda: None)
    flat_chars = len(json.dumps(flat.schemas()))
    hierarchical_chars = len(json.dumps(hierarchical.schemas()))
    assert hierarchical_chars < flat_chars / 5, (
        "Hierarchical main-agent schema should be a fraction of the flat kit's size."
    )


def test_every_task_tool_sequence_is_partitionable_into_delegate_calls():
    """Every tool referenced by the 21 chain tasks must map to a known
    specialist group -- otherwise a delegated sub-task would have nowhere
    to go."""
    tool_names = {t.name for t in ALL_TOOLS}
    for task in _load_all_tasks():
        for name in task["expect_tool_sequence"]:
            assert name in tool_names
            group_name = None
            for g, spec in GROUP_DEFS.items():
                if name.startswith(spec["prefixes"]):
                    group_name = g
                    break
            assert group_name is not None, f"{task['id']}: {name} has no specialist group"


def test_main_system_prompt_names_every_delegate_tool():
    for group_name in GROUP_DEFS:
        assert f"delegate_{group_name}" in MAIN_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Real-LLM hierarchical accuracy (skipped without an API key)
# ---------------------------------------------------------------------------


def _make_live_llm():
    if os.environ.get("DEEPSEEK_API_KEY"):
        from agent import DeepSeekLLM
        return DeepSeekLLM()
    if os.environ.get("BAILIAN_API_KEY"):
        from agent import BailianLLM
        return BailianLLM()
    from agent import OpenAILLM
    return OpenAILLM()


@pytest.mark.skipif(not HAS_LIVE_LLM, reason="No DEEPSEEK_API_KEY/BAILIAN_API_KEY/OPENAI_API_KEY set")
def test_hierarchical_smoke_single_group_task():
    """Coarse smoke gate on one task whose whole chain falls inside a single
    specialist (add_twice: five math_* calls) -- the simplest case, where
    the hierarchy should cost nothing but one delegation. Run
    examples/hierarchical_agent_test.py for the full 21-task report,
    including the harder cross-specialist tasks.
    """
    tasks_by_id = {t["id"]: t for t in _load_all_tasks()}
    task = tasks_by_id["add_twice"]

    call_log = []
    registry = build_main_registry(llm_factory=_make_live_llm, call_log=call_log, max_steps=8)
    agent = ReActAgent(llm=_make_live_llm(), tools=registry, system_prompt=MAIN_SYSTEM_PROMPT, max_steps=6)
    outcome = agent.run(task["prompt"])

    assert any(s.lower() in outcome.answer.lower() for s in task["expect_substrings"]), (
        f"Single-group smoke task did not reach the expected answer: {outcome.answer!r}"
    )
    assert len(call_log) >= 1, "Main agent never delegated to a specialist."
