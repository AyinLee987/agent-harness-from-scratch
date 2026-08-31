"""Tests for the multi-tool-call (chained) accuracy experiment.

Mirrors tests/test_tool_scaling.py's split: deterministic structural checks
run unconditionally; the real-LLM accuracy check is skipped without an API
key, since chained tool *selection* is a model decision MockLLM can't
exercise meaningfully.
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()  # picks up a local .env for the live-LLM test; no-op in CI

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EXAMPLES = os.path.join(_REPO_ROOT, "examples")
sys.path.insert(0, _EXAMPLES)

from agent import ReActAgent, ToolRegistry  # noqa: E402

from tool_scaling_kit import ALL_TOOLS  # noqa: E402

TASKS_PATH = os.path.join(_EXAMPLES, "tool_scaling_multi_tasks.json")

HAS_LIVE_LLM = bool(
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("BAILIAN_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)


def _load_tasks() -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Structural checks (fast, free, always run)
# ---------------------------------------------------------------------------


def test_multi_tasks_reference_only_known_tools():
    tool_names = {t.name for t in ALL_TOOLS}
    for task in _load_tasks():
        for name in task["expect_tool_sequence"]:
            assert name in tool_names, f"{task['id']} references unknown tool {name!r}"


def test_multi_tasks_actually_require_more_than_one_call():
    """Every task in this file should need >= 2 tool calls -- that's the point."""
    for task in _load_tasks():
        assert len(task["expect_tool_sequence"]) >= 2, (
            f"{task['id']} has < 2 expected tool calls; belongs in the single-tool task file."
        )


def test_at_least_one_task_repeats_the_same_tool():
    """The kit should cover 'call the same tool twice' as well as chaining
    different tools -- these stress different failure modes."""
    tasks = _load_tasks()
    assert any(
        len(set(t["expect_tool_sequence"])) < len(t["expect_tool_sequence"])
        for t in tasks
    ), "No multi-task calls the same tool more than once."


def test_chained_outputs_actually_compose_to_the_expected_answer():
    """Execute each task's tool sequence directly (no LLM) with the arguments
    the prompt implies, threading each step's output into the next, and
    check it lands on expect_substrings. This pins down that the *tasks are
    solvable* -- independent of whether a model can find the right sequence.
    """

    tools_by_name = {t.name: t for t in ALL_TOOLS}

    def run_chain(sequence, first_call_kwargs, chain_fn):
        """chain_fn(step_index, prev_output) -> kwargs for that step."""
        output = None
        for i, name in enumerate(sequence):
            kwargs = first_call_kwargs if i == 0 else chain_fn(i, output)
            output = tools_by_name[name].run(**kwargs)
        return output

    checks = [
        ("convert_then_round", {"kilometers": 10}, lambda i, prev: {"value": float(prev), "digits": 1}),
        ("lower_then_vowels", {"text": "Hello World"}, lambda i, prev: {"text": prev}),
        ("add_twice", {"a": 12, "b": 8}, lambda i, prev: {"a": float(prev), "b": 15}),
        ("upper_then_reverse", {"text": "agent"}, lambda i, prev: {"text": prev}),
        ("multiply_twice", {"a": 3, "b": 4}, lambda i, prev: {"a": float(prev), "b": 5}),
        ("base64_roundtrip", {"text": "agent"}, lambda i, prev: {"encoded": prev}),
        ("kg_to_lb_then_round", {"kilograms": 7}, lambda i, prev: {"value": float(prev), "digits": 0}),
        ("average_then_round", {"numbers": "2,3,5,7"}, lambda i, prev: {"value": float(prev), "digits": 1}),
        ("dedupe_then_sort", {"items": "b,a,c,a,b"}, lambda i, prev: {"items": prev}),
        ("add_days_twice", {"date": "2026-01-01", "days": 5}, lambda i, prev: {"date": prev, "days": 20}),
    ]
    tasks_by_id = {t["id"]: t for t in _load_tasks()}
    for task_id, first_kwargs, chain_fn in checks:
        task = tasks_by_id[task_id]
        result = run_chain(task["expect_tool_sequence"], first_kwargs, chain_fn)
        assert any(s.lower() in result.lower() for s in task["expect_substrings"]), (
            f"{task_id}: chained result {result!r} missing any of {task['expect_substrings']}"
        )


# ---------------------------------------------------------------------------
# Real-LLM multi-tool-call accuracy (skipped without an API key)
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
def test_multi_tool_sequence_accuracy_smoke():
    """Coarse smoke gate on a sample of chained tasks against the full
    50-tool registry. Run examples/tool_scaling_multi_test.py for the full
    14-task report.
    """

    tasks_by_id = {t["id"]: t for t in _load_tasks()}
    sample_ids = ["add_twice", "convert_then_round", "base64_roundtrip"]
    registry = ToolRegistry(ALL_TOOLS)

    correct = 0
    for task_id in sample_ids:
        task = tasks_by_id[task_id]
        agent = ReActAgent(llm=_make_live_llm(), tools=registry, max_steps=8)
        outcome = agent.run(task["prompt"])
        called = [(s.get("action") or {}).get("name") for s in outcome.trajectory if s.get("action")]
        correct += int(called == task["expect_tool_sequence"])

    accuracy = correct / len(sample_ids)
    assert accuracy >= 1 / 3, (
        f"Multi-tool-call accuracy was {accuracy:.0%} on the smoke sample, "
        f"below the floor (at least 1 of {len(sample_ids)} expected)."
    )
