"""Tests for the verbose-description variant of the 50-tool kit.

Deterministic checks (always run): the verbose kit is a faithful wrapper --
same names, same schemas, same run() behavior -- with meaningfully longer
descriptions, so any accuracy difference against the concise kit is
attributable to description length/repetition alone. The live-LLM checks
are skip-gated on an API key, same pattern as the other tool_scaling test
files.
"""

from __future__ import annotations

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
from tool_scaling_verbose_kit import ALL_TOOLS_VERBOSE  # noqa: E402

HAS_LIVE_LLM = bool(
    os.environ.get("DEEPSEEK_API_KEY")
    or os.environ.get("BAILIAN_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
)


def test_verbose_kit_has_same_names_as_concise_kit():
    assert {t.name for t in ALL_TOOLS_VERBOSE} == {t.name for t in ALL_TOOLS}
    assert len(ALL_TOOLS_VERBOSE) == 50


def test_verbose_descriptions_are_substantially_longer():
    concise_by_name = {t.name: t for t in ALL_TOOLS}
    for verbose_tool in ALL_TOOLS_VERBOSE:
        concise_len = len(concise_by_name[verbose_tool.name].description)
        verbose_len = len(verbose_tool.description)
        assert verbose_len >= concise_len * 3, (
            f"{verbose_tool.name}: verbose description ({verbose_len} chars) "
            f"is not meaningfully longer than concise ({concise_len} chars)."
        )


def test_verbose_wrapper_preserves_schema_and_behavior():
    """Only description should differ -- parameters and run() must match."""

    concise_by_name = {t.name: t for t in ALL_TOOLS}
    for verbose_tool in ALL_TOOLS_VERBOSE:
        concise_tool = concise_by_name[verbose_tool.name]
        assert verbose_tool.parameters_schema() == concise_tool.parameters_schema()
    # Spot-check run() delegates correctly for a representative sample.
    verbose_by_name = {t.name: t for t in ALL_TOOLS_VERBOSE}
    assert verbose_by_name["math_add"].run(a=23, b=19) == "42"
    assert verbose_by_name["text_reverse"].run(text="hello") == "olleh"
    assert verbose_by_name["data_base64_encode"].run(text="hello") == "aGVsbG8="


def test_full_registry_schema_is_at_least_5x_larger():
    import json
    concise_chars = len(json.dumps([t.to_schema() for t in ALL_TOOLS]))
    verbose_chars = len(json.dumps([t.to_schema() for t in ALL_TOOLS_VERBOSE]))
    assert verbose_chars >= concise_chars * 5


@pytest.mark.skipif(not HAS_LIVE_LLM, reason="No DEEPSEEK_API_KEY/BAILIAN_API_KEY/OPENAI_API_KEY set")
def test_tool_selection_still_works_with_verbose_descriptions_smoke():
    """Coarse smoke gate: a couple of tasks should still resolve correctly
    even with ~6.5x longer, repetitive tool descriptions. Run
    examples/tool_scaling_verbose_test.py for the full comparison.
    """

    if os.environ.get("DEEPSEEK_API_KEY"):
        from agent import DeepSeekLLM
        llm_factory = DeepSeekLLM
    elif os.environ.get("BAILIAN_API_KEY"):
        from agent import BailianLLM
        llm_factory = BailianLLM
    else:
        from agent import OpenAILLM
        llm_factory = OpenAILLM

    registry = ToolRegistry(ALL_TOOLS_VERBOSE)
    agent = ReActAgent(llm=llm_factory(), tools=registry, max_steps=5)
    outcome = agent.run("What is 23 times 17?")
    used = any((s.get("action") or {}).get("name") == "math_multiply" for s in outcome.trajectory)
    assert used, "math_multiply was not selected even in a 2-tool-equivalent smoke case."
