"""Tests for retrieval-based tool selection."""

from __future__ import annotations

import pytest

from agent import (
    AllToolsSelector,
    LexicalToolSelector,
    MockLLM,
    ReActAgent,
    ToolRegistry,
    tool,
)
from agent.llm import LLMResponse, ToolCall, Usage
from agent.trigger.tool_router import filtered_schemas


def _registry(count: int = 20) -> ToolRegistry:
    """A registry with a few meaningfully-named tools plus filler."""

    @tool
    def weather_forecast(city: str) -> str:
        """Look up the weather forecast for a city."""
        return f"sunny in {city}"

    @tool
    def currency_convert(amount: float, currency: str) -> str:
        """Convert an amount of money between currencies."""
        return "42"

    @tool
    def spawn_subagent(role: str, task: str) -> str:
        """Delegate a sub-task to a specialist worker."""
        return "spawned"

    tools = [weather_forecast, currency_convert, spawn_subagent]
    for index in range(count - len(tools)):
        def _filler(text: str, _index: int = index) -> str:
            return text
        _filler.__doc__ = f"Unrelated placeholder operation number {index}."
        _filler.__name__ = f"placeholder_{index}"
        tools.append(tool(_filler))
    return ToolRegistry(tools)


# -- selection --------------------------------------------------------------
def test_a_small_registry_is_never_filtered():
    """Filtering six tools saves nothing worth the risk of hiding the right
    one, so selection is a no-op below min_tools."""

    registry = _registry(6)
    selection = LexicalToolSelector(top_k=2, min_tools=12).select(
        registry, "what is the weather in Beijing"
    )

    assert selection.filtered is False
    assert set(selection.names) == set(registry.names())


def test_the_matching_tool_is_offered_and_the_filler_is_not():
    registry = _registry(20)
    selection = LexicalToolSelector(top_k=3, min_tools=5, pinned=[]).select(
        registry, "what is the weather forecast in Beijing"
    )

    assert "weather_forecast" in selection.names
    assert selection.filtered
    assert len(selection.names) <= 3


def test_pinned_tools_survive_a_query_that_does_not_mention_them():
    """No task text lexically matches "spawn_subagent", but hiding it
    silently removes the agent's ability to delegate."""

    registry = _registry(20)
    selection = LexicalToolSelector(
        top_k=2, min_tools=5, pinned=["spawn_subagent"]
    ).select(registry, "convert 100 dollars to euros")

    assert "spawn_subagent" in selection.names
    assert "currency_convert" in selection.names


def test_a_tool_already_used_is_never_hidden_from_a_later_step():
    """Dropping a tool mid-chain -- after the model has seen its output and
    may reason about calling it again -- produces an "unknown tool" retry
    loop rather than a cheaper prompt."""

    registry = _registry(20)
    selection = LexicalToolSelector(top_k=2, min_tools=5, pinned=[]).select(
        registry,
        "what is the weather in Beijing",
        used=["currency_convert"],
    )

    assert "currency_convert" in selection.names


def test_a_pinned_or_used_name_that_is_not_registered_is_ignored():
    registry = _registry(20)
    selection = LexicalToolSelector(
        top_k=2, min_tools=5, pinned=["not_a_real_tool"]
    ).select(registry, "weather", used=["also_not_real"])

    assert "not_a_real_tool" not in selection.names
    assert "also_not_real" not in selection.names


def test_a_query_that_matches_nothing_offers_everything():
    """Regression guard, found by running the selector against this repo's
    own live 56-tool registry: a Chinese question scored 0.0 on every
    (English-described) tool, and `sorted` happily returned the first
    top_k in registry order -- an arbitrary slice presented as a ranking,
    identical for every unrelated question. A selector with no opinion must
    not act like it has one."""

    registry = _registry(20)
    selector = LexicalToolSelector(top_k=3, min_tools=5, pinned=[])

    first = selector.select(registry, "23 乘以 17 等于多少")
    second = selector.select(registry, "把这段文字转成大写")

    assert first.filtered is False
    assert len(first.names) == 20
    # And crucially it is not just "the same 3 tools for both questions".
    assert first.names == second.names == registry.names()


def test_matching_still_narrows_when_the_query_shares_the_descriptions_language():
    registry = _registry(20)
    selector = LexicalToolSelector(top_k=3, min_tools=5, pinned=[])

    assert "weather_forecast" in selector.select(
        registry, "what is the weather forecast"
    ).names
    assert "currency_convert" in selector.select(
        registry, "convert money between currencies"
    ).names


def test_the_offered_set_is_never_padded_with_zero_scoring_tools():
    registry = _registry(20)
    selection = LexicalToolSelector(top_k=8, min_tools=5, pinned=[]).select(
        registry, "weather"
    )

    # Only one tool mentions weather, so exactly one should be offered --
    # not eight, seven of which matched nothing.
    assert selection.names == ["weather_forecast"]


def test_all_tools_selector_offers_everything():
    registry = _registry(20)
    selection = AllToolsSelector().select(registry, "anything")

    assert selection.filtered is False
    assert len(selection.names) == 20


def test_top_k_must_be_positive():
    with pytest.raises(ValueError):
        LexicalToolSelector(top_k=0)


def test_filtered_schemas_keeps_registry_order():
    registry = _registry(20)
    selection = LexicalToolSelector(top_k=5, min_tools=5, pinned=[]).select(
        registry, "weather forecast currency"
    )
    schemas = filtered_schemas(registry, selection)

    names = [item["function"]["name"] for item in schemas]
    assert names == [n for n in registry.names() if n in set(selection.names)]


# -- ReActLoop integration --------------------------------------------------
def test_the_loop_offers_the_model_only_the_selected_tools():
    registry = _registry(20)
    seen = []

    class _RecordingLLM(MockLLM):
        def chat(self, messages, tools=None):
            seen.append([t["function"]["name"] for t in (tools or [])])
            return LLMResponse(content="done", usage=Usage(1, 1))

    ReActAgent(
        llm=_RecordingLLM(),
        tools=registry,
        tool_selector=LexicalToolSelector(top_k=2, min_tools=5, pinned=[]),
    ).run("what is the weather forecast in Beijing")

    assert seen
    assert len(seen[0]) <= 2
    assert "weather_forecast" in seen[0]


def test_no_selector_still_offers_every_tool():
    """The default must stay exactly what it was before tool routing existed."""

    registry = _registry(20)
    seen = []

    class _RecordingLLM(MockLLM):
        def chat(self, messages, tools=None):
            seen.append([t["function"]["name"] for t in (tools or [])])
            return LLMResponse(content="done", usage=Usage(1, 1))

    ReActAgent(llm=_RecordingLLM(), tools=registry).run("anything")

    assert len(seen[0]) == 20


def test_the_selection_query_grows_with_the_trajectory():
    """A chain whose later step needs a tool the original question never
    mentioned must still be able to see it."""

    from agent.state.context import ExecutionContext
    from agent.trigger.react_loop import _selection_query, _tools_used

    ctx = ExecutionContext()
    ctx.add_message("system", "sys")
    ctx.add_message("user", "fetch the page")
    step = ctx.new_step()
    step.thought = "I should look at the exchange rate"
    step.record_tool_call(
        id="c1",
        name="currency_convert",
        arguments={},
        observation="1 USD = 7.1 CNY",
        ok=True,
    )

    query = _selection_query(ctx)
    assert "fetch the page" in query
    assert "exchange rate" in query
    assert "7.1 CNY" in query
    assert _tools_used(ctx) == ["currency_convert"]
