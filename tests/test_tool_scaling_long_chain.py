"""Tests for the long tool-call chain (5+ steps) accuracy experiment.

Same split as the other tool_scaling test files: deterministic structural
checks (and a no-LLM chain-composition check) always run; the live-LLM
accuracy check is skip-gated on an API key.
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

from agent import ReActAgent, ToolRegistry  # noqa: E402

from tool_scaling_kit import ALL_TOOLS  # noqa: E402

TASKS_PATH = os.path.join(_EXAMPLES, "tool_scaling_long_chain_tasks.json")

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


def test_long_chain_tasks_reference_only_known_tools():
    tool_names = {t.name for t in ALL_TOOLS}
    for task in _load_tasks():
        for name in task["expect_tool_sequence"]:
            assert name in tool_names, f"{task['id']} references unknown tool {name!r}"


def test_every_chain_is_at_least_five_steps():
    """Every task in this file must be a 5+ step chain -- short chains
    belong in the single-tool or (now similarly long) multi-task files."""
    for task in _load_tasks():
        length = len(task["expect_tool_sequence"])
        assert length >= 5, f"{task['id']} has only {length} expected tool calls; needs >= 5."


def test_chained_outputs_actually_compose_to_the_expected_answer():
    """Execute each task's tool sequence directly (no LLM), threading
    outputs through the chain with the arguments the prompt implies, and
    check the trajectory lands on expect_substrings. Confirms the *tasks
    are solvable* independent of whether a model can find the right
    sequence. Checked against every step's output (not just the last) since
    a real LLM's final answer summarizes the whole trajectory.
    """

    tools_by_name = {t.name: t for t in ALL_TOOLS}

    def run_chain(sequence, step_kwargs):
        """step_kwargs(step_index, outputs_so_far: list[str]) -> kwargs for that step."""
        outputs: list[str] = []
        for i, name in enumerate(sequence):
            kwargs = step_kwargs(i, outputs)
            outputs.append(tools_by_name[name].run(**kwargs))
        return outputs

    checks = {
        "sum_div_round": lambda i, o: (
            {"numbers": "3,7,2,9,4"} if i == 0
            else {"a": float(o[0]), "b": 5} if i == 1
            else {"value": float(o[1]), "digits": 1} if i == 2
            else {"a": float(o[2]), "b": 3} if i == 3
            else {"a": float(o[3]), "b": 2}
        ),
        "snake_reverse_vowels": lambda i, o: (
            {"text": "Hello World"} if i == 0
            else {"text": o[0]} if i == 1
            else {"text": o[1]} if i == 2
            else {"a": float(o[2]), "b": 4} if i == 3
            else {"value": float(o[3]), "digits": 0}
        ),
        # Branches: steps 2-4 all key off the date computed at step 0
        # (o[0]), not each other's output.
        "add_days_weekday_weekend": lambda i, o: (
            {"date": "2026-01-01", "days": 45} if i == 0
            else {"date": o[0]} if i == 1
            else {"date": o[0]} if i == 2
            else {"text": o[2]} if i == 3
            else {"text": o[3]}
        ),
        "km_mi_nm_round": lambda i, o: (
            {"kilometers": 10} if i == 0
            else {"value": float(o[0]), "digits": 2} if i == 1
            else {"miles": float(o[1])} if i == 2
            else {"value": float(o[2]), "digits": 1} if i == 3
            else {"nautical_miles": float(o[3])}
        ),
        "base64_hex_roundtrip": lambda i, o: (
            {"text": "chain"} if i == 0
            else {"text": o[0]} if i == 1
            else {"hex_string": o[1]} if i == 2
            else {"encoded": o[2]} if i == 3
            else {"text": o[3]}
        ),
        "sum_div_round_temp_round": lambda i, o: (
            {"numbers": "4,8,15,16,23"} if i == 0
            else {"a": float(o[0]), "b": 5} if i == 1
            else {"value": float(o[1]), "digits": 1} if i == 2
            else {"celsius": float(o[2])} if i == 3
            else {"value": float(o[3]), "digits": 0}
        ),
        "lower_snake_reverse_count_leap": lambda i, o: (
            {"text": "Data Science Rocks"} if i == 0
            else {"text": o[0]} if i == 1
            else {"text": o[1]} if i == 2
            else {"text": o[2]} if i == 3
            else {"year": int(o[3])}
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
# Real-LLM long-chain accuracy (skipped without an API key)
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
def test_long_chain_final_answer_smoke():
    """Coarse smoke gate on one 5-step task, checking only the final answer
    (not exact sequence, which is the harder metric the full sweep in
    examples/tool_scaling_long_chain_test.py reports on).
    """

    tasks_by_id = {t["id"]: t for t in _load_tasks()}
    task = tasks_by_id["snake_reverse_vowels"]
    registry = ToolRegistry(ALL_TOOLS)
    agent = ReActAgent(llm=_make_live_llm(), tools=registry, max_steps=14)
    outcome = agent.run(task["prompt"])
    assert any(s.lower() in outcome.answer.lower() for s in task["expect_substrings"]), (
        f"5-step chain smoke task did not reach the expected answer: {outcome.answer!r}"
    )
