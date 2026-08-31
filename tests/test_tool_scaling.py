"""Tests for the 50-tool scaling kit (examples/tool_scaling_kit.py).

Two layers:

1. Deterministic, free, always-run: the kit itself is well-formed (50 unique
   tools, valid schemas) and each tool's ``run()`` produces the exact
   expected output -- this is ordinary unit testing of the new tools.
2. Real-LLM tool-*selection* accuracy: skipped unless an API key is
   configured, since MockLLM's tool choice is keyword-heuristic, not a model
   decision, and can't exercise the thing we actually want to measure (see
   examples/tool_scaling_test.py for the full sweep across registry sizes;
   this is a fast, cheap sanity check that the wiring works end to end).
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

from tool_scaling_kit import ALL_TOOLS, build_registry  # noqa: E402

TASKS_PATH = os.path.join(_EXAMPLES, "tool_scaling_tasks.json")

HAS_LIVE_LLM = bool(
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("BAILIAN_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)


def _load_tasks() -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Layer 1: kit structure + tool correctness (fast, free, always run)
# ---------------------------------------------------------------------------


def test_kit_has_exactly_50_uniquely_named_tools():
    assert len(ALL_TOOLS) == 50
    assert len({t.name for t in ALL_TOOLS}) == 50


def test_every_tool_has_a_description_and_valid_schema():
    for t in ALL_TOOLS:
        schema = t.to_schema()
        assert schema["function"]["name"] == t.name
        assert schema["function"]["description"], f"{t.name} has no description"
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        # every declared parameter must carry a JSON-schema type
        for pname, pschema in params["properties"].items():
            assert "type" in pschema, f"{t.name}.{pname} missing a schema type"


def test_build_registry_takes_a_stable_prefix_slice():
    reg10 = build_registry(10)
    assert reg10.names() == [t.name for t in ALL_TOOLS[:10]]
    with pytest.raises(ValueError):
        build_registry(0)
    with pytest.raises(ValueError):
        build_registry(51)


def test_tasks_file_covers_every_tool_exactly_once_in_matching_order():
    tasks = _load_tasks()
    assert len(tasks) == 50
    assert [t["expect_tool"] for t in tasks] == [tool.name for tool in ALL_TOOLS]


# Direct golden-value checks for representative tools spanning every
# category. These pin down behavior the LLM-facing tasks above only check
# indirectly (via substring matching on a model's paraphrased answer).
@pytest.mark.parametrize(
    "name,kwargs,expected",
    [
        ("math_add", {"a": 23, "b": 19}, "42"),
        ("math_divide", {"a": 144, "b": 12}, "12"),
        ("math_divide", {"a": 1, "b": 0}, "Cannot divide by zero."),
        ("math_average", {"numbers": "2,4,6"}, "4"),
        ("text_reverse", {"text": "hello"}, "olleh"),
        ("text_is_palindrome", {"text": "racecar"}, "true"),
        ("text_is_palindrome", {"text": "hello"}, "false"),
        ("date_day_of_week", {"date": "2026-08-31"}, "Monday"),
        ("date_quarter", {"date": "2026-08-31"}, "Q3"),
        ("convert_celsius_to_fahrenheit", {"celsius": 100}, "212"),
        ("convert_fahrenheit_to_celsius", {"fahrenheit": 212}, "100"),
        ("data_base64_encode", {"text": "hello"}, "aGVsbG8="),
        ("data_base64_decode", {"encoded": "aGVsbG8="}, "hello"),
        ("data_dedupe_list", {"items": "a,b,a,c,b"}, "a,b,c"),
    ],
)
def test_representative_tools_produce_exact_expected_output(name, kwargs, expected):
    tools_by_name = {t.name: t for t in ALL_TOOLS}
    assert tools_by_name[name].run(**kwargs) == expected


# ---------------------------------------------------------------------------
# Layer 2: real-LLM tool-selection sanity check (skipped without an API key)
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
def test_tool_selection_accuracy_with_full_50_tool_registry():
    """Smoke check: with all 50 tools registered, a small sample of tasks
    should still resolve to the *correct* tool most of the time. This is a
    coarse pass/fail gate, not the full sweep -- run
    ``examples/tool_scaling_test.py`` for the size-by-size accuracy curve.
    """

    tasks = _load_tasks()
    sample_ids = {
        "math_multiply", "text_replace", "date_add_days",
        "convert_km_to_mi", "data_json_validate",
    }
    sample = [t for t in tasks if t["id"] in sample_ids]
    registry = ToolRegistry(ALL_TOOLS)

    correct = 0
    for task in sample:
        agent = ReActAgent(llm=_make_live_llm(), tools=registry, max_steps=5)
        outcome = agent.run(task["prompt"])
        used = any(
            (step.get("action") or {}).get("name") == task["expect_tool"]
            for step in outcome.trajectory
        )
        correct += int(used)

    accuracy = correct / len(sample)
    # A generous floor: this is a smoke test, not the accuracy benchmark.
    assert accuracy >= 0.6, (
        f"Tool-selection accuracy with 50 tools registered was {accuracy:.0%}, "
        f"below the 60% smoke-test floor."
    )
