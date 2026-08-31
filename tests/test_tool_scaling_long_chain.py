"""Tests for the long tool-call chain (3-5 steps) accuracy experiment.

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


def test_chain_lengths_span_three_to_five_and_include_each():
    lengths = sorted(len(t["expect_tool_sequence"]) for t in _load_tasks())
    assert min(lengths) >= 3, "This file is for chains of 3+ steps -- shorter belongs in the multi-task file."
    assert set(lengths) >= {3, 4, 5}, f"Expected chain lengths 3, 4, and 5 to each be covered; got {lengths}"


def test_chained_outputs_actually_compose_to_the_expected_answer():
    """Execute each task's tool sequence directly (no LLM), threading each
    step's output into the next with the arguments the prompt implies, and
    check it lands on expect_substrings. Confirms the *tasks are solvable*
    independent of whether a model can find the right sequence.
    """

    tools_by_name = {t.name: t for t in ALL_TOOLS}

    def run_chain(sequence, steps_kwargs):
        output = None
        for i, name in enumerate(sequence):
            kwargs = steps_kwargs(i, output)
            output = tools_by_name[name].run(**kwargs)
        return output

    checks = {
        "sum_div_round": lambda i, prev: (
            {"numbers": "3,7,2,9,4"} if i == 0
            else {"a": float(prev), "b": 5} if i == 1
            else {"value": float(prev), "digits": 1}
        ),
        "snake_reverse_vowels": lambda i, prev: (
            {"text": "Hello World"} if i == 0 else {"text": prev}
        ),
        "add_days_weekday_weekend": lambda i, prev: (
            {"date": "2026-01-01", "days": 45} if i == 0 else {"date": prev}
        ),
        "km_mi_nm_round": lambda i, prev: (
            {"kilometers": 10} if i == 0
            else {"value": float(prev), "digits": 2} if i == 1
            else {"miles": float(prev)} if i == 2
            else {"value": float(prev), "digits": 1}
        ),
        "base64_hex_roundtrip": lambda i, prev: (
            {"text": "chain"} if i == 0
            else {"text": prev} if i == 1
            else {"hex_string": prev} if i == 2
            else {"encoded": prev}
        ),
        "sum_div_round_temp_round": lambda i, prev: (
            {"numbers": "4,8,15,16,23"} if i == 0
            else {"a": float(prev), "b": 5} if i == 1
            else {"value": float(prev), "digits": 1} if i == 2
            else {"celsius": float(prev)} if i == 3
            else {"value": float(prev), "digits": 0}
        ),
        "lower_snake_reverse_count_leap": lambda i, prev: (
            {"text": "Data Science Rocks"} if i == 0
            else {"text": prev} if i in (1, 2)
            else {"text": prev} if i == 3
            else {"year": int(prev)}
        ),
    }

    tasks_by_id = {t["id"]: t for t in _load_tasks()}
    for task_id, chain_fn in checks.items():
        task = tasks_by_id[task_id]
        result = run_chain(task["expect_tool_sequence"], chain_fn)
        assert any(s.lower() in result.lower() for s in task["expect_substrings"]), (
            f"{task_id}: chained result {result!r} missing any of {task['expect_substrings']}"
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
    """Coarse smoke gate on the shortest (3-step) task, checking only the
    final answer (not exact sequence, which is the harder metric the full
    sweep in examples/tool_scaling_long_chain_test.py reports on).
    """

    tasks_by_id = {t["id"]: t for t in _load_tasks()}
    task = tasks_by_id["snake_reverse_vowels"]
    registry = ToolRegistry(ALL_TOOLS)
    agent = ReActAgent(llm=_make_live_llm(), tools=registry, max_steps=12)
    outcome = agent.run(task["prompt"])
    assert any(s.lower() in outcome.answer.lower() for s in task["expect_substrings"]), (
        f"3-step chain smoke task did not reach the expected answer: {outcome.answer!r}"
    )
