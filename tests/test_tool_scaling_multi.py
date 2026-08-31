"""Tests for the multi-tool-call (chained) accuracy experiment.

Every task in tool_scaling_multi_tasks.json is a 5+ step chain (originally
2 steps; extended so every "short chain" case is a long one, per the same
reasoning as tool_scaling_long_chain_tasks.json -- longer chains give
errors more opportunities to compound).

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


def test_multi_tasks_require_at_least_five_calls():
    """Every task in this file should need >= 5 tool calls -- long chains are
    a stricter probe than short ones: each extra correct pick compounds the
    chance of an error, so this file only carries long chains now."""
    for task in _load_tasks():
        assert len(task["expect_tool_sequence"]) >= 5, (
            f"{task['id']} has only {len(task['expect_tool_sequence'])} expected tool "
            f"calls; every task here must be a 5+ step chain."
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
    the prompt implies, threading outputs through the chain, and check the
    last step's result lands on expect_substrings. This pins down that the
    *tasks are solvable* -- independent of whether a model can find the
    right sequence. Every chain here is 5+ steps; a couple (e.g.
    add_days_twice) branch -- later steps reuse an earlier output (the
    computed date) rather than always the immediately preceding one -- so
    each step function sees the *full list* of outputs so far, not just the
    last one.
    """

    tools_by_name = {t.name: t for t in ALL_TOOLS}

    def run_chain(sequence, step_kwargs):
        """step_kwargs(step_index, outputs_so_far: list[str]) -> kwargs for that step.

        Returns every step's output, not just the last: a real LLM's final
        answer summarizes the whole trajectory (e.g. "it's Monday, not a
        weekend, ISO week 5"), not just the literal return value of the
        final tool call -- branching tasks like add_days_twice have their
        checkable fact land on an earlier step, not the last one.
        """
        outputs: list[str] = []
        for i, name in enumerate(sequence):
            kwargs = step_kwargs(i, outputs)
            outputs.append(tools_by_name[name].run(**kwargs))
        return outputs

    checks = {
        "convert_then_round": lambda i, o: (
            {"kilometers": 10} if i == 0
            else {"value": float(o[0]), "digits": 2} if i == 1
            else {"miles": float(o[1])} if i == 2
            else {"value": float(o[2]), "digits": 1} if i == 3
            else {"a": float(o[3]), "b": 1}
        ),
        "lower_then_vowels": lambda i, o: (
            {"text": "Hello World"} if i == 0
            else {"text": o[0]} if i == 1
            else {"a": float(o[1]), "b": 10} if i == 2
            else {"a": float(o[2]), "b": 2} if i == 3
            else {"value": float(o[3]), "digits": 0}
        ),
        "add_days_then_weekday": lambda i, o: (
            {"date": "2026-08-31", "days": 10} if i == 0
            else {"date": o[0]} if i == 1
            else {"text": o[1]} if i == 2
            else {"text": o[2]} if i == 3
            else {"text": o[3]}
        ),
        "celsius_roundtrip": lambda i, o: (
            {"celsius": 100} if i == 0
            else {"fahrenheit": float(o[0])} if i == 1
            else {"celsius": float(o[1])} if i == 2
            else {"fahrenheit": float(o[2])} if i == 3
            else {"value": float(o[3]), "digits": 0}
        ),
        "add_twice": lambda i, o: (
            {"a": 12, "b": 8} if i == 0
            else {"a": float(o[0]), "b": 15} if i == 1
            else {"a": float(o[1]), "b": 2} if i == 2
            else {"value": float(o[2]), "digits": 0} if i == 3
            else {"a": float(o[3]), "b": 7}
        ),
        "upper_then_reverse": lambda i, o: (
            {"text": "agent"} if i == 0
            else {"text": o[0]} if i == 1
            else {"text": o[1]} if i == 2
            else {"a": float(o[2]), "b": 3} if i == 3
            else {"value": float(o[3]), "digits": 0}
        ),
        "multiply_twice": lambda i, o: (
            {"a": 3, "b": 4} if i == 0
            else {"a": float(o[0]), "b": 5} if i == 1
            else {"a": float(o[1]), "b": 10} if i == 2
            else {"a": float(o[2]), "b": 20} if i == 3
            else {"value": float(o[3]), "digits": 0}
        ),
        "base64_roundtrip": lambda i, o: (
            {"text": "chain"} if i == 0
            else {"text": o[0]} if i == 1
            else {"hex_string": o[1]} if i == 2
            else {"encoded": o[2]} if i == 3
            else {"text": o[3]}
        ),
        "add_days_then_days_between": lambda i, o: (
            {"date": "2026-01-01", "days": 100} if i == 0
            else {"start_date": o[0], "end_date": "2026-12-31"} if i == 1
            else {"a": float(o[1]), "b": 2} if i == 2
            else {"value": float(o[2]), "digits": 1} if i == 3
            else {"a": float(o[3]), "b": 8}
        ),
        "kg_to_lb_then_round": lambda i, o: (
            {"kilograms": 7} if i == 0
            else {"value": float(o[0]), "digits": 0} if i == 1
            else {"pounds": float(o[1])} if i == 2
            else {"value": float(o[2]), "digits": 1} if i == 3
            else {"a": float(o[3]), "b": 2}
        ),
        "replace_then_wordcount": lambda i, o: (
            {"text": "I love cats and cats love me", "old": "cats", "new": "dogs"} if i == 0
            else {"text": o[0]} if i == 1
            else {"a": float(o[1]), "b": 3} if i == 2
            else {"a": float(o[2]), "b": 1} if i == 3
            else {"value": float(o[3]), "digits": 0}
        ),
        "average_then_round": lambda i, o: (
            {"numbers": "2,3,5,7"} if i == 0
            else {"value": float(o[0]), "digits": 1} if i == 1
            else {"celsius": float(o[1])} if i == 2
            else {"value": float(o[2]), "digits": 0} if i == 3
            else {"a": float(o[3]), "b": 5}
        ),
        "dedupe_then_sort": lambda i, o: (
            {"items": "b,a,c,a,b"} if i == 0
            else {"items": o[0], "descending": False} if i == 1
            else {"items": o[1], "descending": True} if i == 2
            else {"text": o[2]} if i == 3
            else {"text": o[3]}
        ),
        # Branches: steps 2-4 all key off the date computed at step 1
        # (o[1]), not each other's output.
        "add_days_twice": lambda i, o: (
            {"date": "2026-01-01", "days": 5} if i == 0
            else {"date": o[0], "days": 20} if i == 1
            else {"date": o[1]}
        ),
    }
    tasks_by_id = {t["id"]: t for t in _load_tasks()}
    assert set(checks) == set(tasks_by_id), (
        f"checks covers {set(checks)} but the tasks file has {set(tasks_by_id)} -- "
        f"every task needs a composition check."
    )
    for task_id, step_kwargs in checks.items():
        task = tasks_by_id[task_id]
        outputs = run_chain(task["expect_tool_sequence"], step_kwargs)
        joined = " ".join(outputs).lower()
        assert any(s.lower() in joined for s in task["expect_substrings"]), (
            f"{task_id}: chained outputs {outputs!r} missing any of {task['expect_substrings']}"
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
    """Coarse smoke gate on a sample of 5-step chained tasks against the
    full tool registry. Run examples/tool_scaling_multi_test.py for the
    full 14-task report.
    """

    tasks_by_id = {t["id"]: t for t in _load_tasks()}
    sample_ids = ["add_twice", "convert_then_round", "base64_roundtrip"]
    registry = ToolRegistry(ALL_TOOLS)

    correct = 0
    for task_id in sample_ids:
        task = tasks_by_id[task_id]
        agent = ReActAgent(llm=_make_live_llm(), tools=registry, max_steps=14)
        outcome = agent.run(task["prompt"])
        called = [(s.get("action") or {}).get("name") for s in outcome.trajectory if s.get("action")]
        correct += int(called == task["expect_tool_sequence"])

    accuracy = correct / len(sample_ids)
    assert accuracy >= 1 / 3, (
        f"Multi-tool-call accuracy was {accuracy:.0%} on the smoke sample, "
        f"below the floor (at least 1 of {len(sample_ids)} expected)."
    )
