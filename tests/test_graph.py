"""Tests for the StateGraph engine (agent/graph.py)."""

import pytest

from agent.trigger.graph import StateGraph


# -- helper nodes for testing ------------------------------------------------
def _echo_node(state: dict) -> dict:
    """Node that appends its own name to a visited list."""
    name = state.get("__current_node__", "echo")
    state.setdefault("visited", []).append(name)
    state["__next__"] = "next"
    return state


def _terminal_node(state: dict) -> dict:
    """Node that signals the end."""
    state.setdefault("visited", []).append("terminal")
    state["__next__"] = "finish"
    return state


def _counter_node(state: dict) -> dict:
    """Node that increments a counter and stops after 3 visits."""
    count = state.setdefault("count", 0) + 1
    state["count"] = count
    state.setdefault("visited", []).append(f"counter_{count}")
    if count >= 3:
        state["__next__"] = "finish"
    else:
        state["__next__"] = "counter"
    return state


def _conditional_node(state: dict) -> dict:
    """Node that routes based on a ``route`` key in state."""
    state.setdefault("visited", []).append("conditional")
    # Route is controlled by the test via state["route"]
    state["__next__"] = state.get("route", "finish")
    return state


def _value_setter_node(value: str):
    """Factory: returns a node that sets a value in state."""

    def node(state: dict) -> dict:
        state["value"] = value
        state["__next__"] = "finish"
        return state

    return node


def _failing_node(state: dict) -> dict:
    """Node that raises an exception."""
    raise RuntimeError("node failure")


# -- tests -------------------------------------------------------------------
class TestGraphConstruction:
    """Tests for the builder API."""

    def test_add_node_and_entry(self):
        g = StateGraph()
        g.add_node("start", _echo_node)
        g.set_entry_point("start")
        assert g._entry == "start"
        assert "start" in g._nodes

    def test_add_node_empty_name_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="non-empty"):
            g.add_node("", _echo_node)

    def test_add_node_reserved_name_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="reserved"):
            g.add_node("__end__", _echo_node)

    def test_set_entry_point_unknown_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="not a registered node"):
            g.set_entry_point("nonexistent")

    def test_add_edge_unknown_source_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="not registered"):
            g.add_edge("nonexistent", "__end__")

    def test_add_conditional_edges_unknown_source_raises(self):
        g = StateGraph()
        with pytest.raises(ValueError, match="not registered"):
            g.add_conditional_edges("nonexistent", lambda s: "x", {"x": "y"})

    def test_add_conditional_edges_empty_mapping_raises(self):
        g = StateGraph()
        g.add_node("a", _echo_node)
        with pytest.raises(ValueError, match="non-empty"):
            g.add_conditional_edges("a", lambda s: "x", {})

    def test_compile_without_entry_raises(self):
        g = StateGraph()
        g.add_node("a", _echo_node)
        with pytest.raises(ValueError, match="No entry point"):
            g.compile()

    def test_chaining(self):
        g = (
            StateGraph()
            .add_node("a", _echo_node)
            .add_node("b", _terminal_node)
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "__end__")
        )
        executor = g.compile()
        state = executor({"__current_node__": "a"})
        assert "a" in state["visited"]
        # _terminal_node pushes "terminal", not the node name "b".
        assert "terminal" in state["visited"]


class TestFixedEdges:
    """Tests for graphs with only fixed (unconditional) edges."""

    def test_simple_chain(self):
        g = StateGraph()
        g.add_node("a", lambda s: {**s, "a_done": True, "__next__": "b"})
        g.add_node("b", lambda s: {**s, "b_done": True, "__next__": "finish"})
        g.set_entry_point("a")
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        executor = g.compile()
        state = executor({})
        assert state["a_done"] is True
        assert state["b_done"] is True

    def test_fixed_edge_overrides_next(self):
        """A fixed edge ignores the __next__ value set by the node."""
        g = StateGraph()
        g.add_node("a", lambda s: {**s, "__next__": "finish"})  # wants to finish
        g.add_node("b", lambda s: {**s, "b_visited": True, "__next__": "finish"})
        g.set_entry_point("a")
        # Fixed edge a→b: even though a sets __next__="finish", the engine
        # follows the fixed edge to b.
        g.add_edge("a", "b")
        g.add_edge("b", "__end__")
        executor = g.compile()
        state = executor({})
        assert state.get("b_visited") is True


class TestConditionalEdges:
    """Tests for graphs with conditional (router-based) edges."""

    def test_conditional_routing(self):
        g = StateGraph()
        g.add_node("think", _conditional_node)
        g.add_node("tools", lambda s: {**s, "tool_executed": True, "__next__": "think"})
        g.set_entry_point("think")
        g.add_conditional_edges(
            "think",
            lambda s: s.get("__next__", "finish"),
            {"tools": "tools", "finish": "__end__"},
        )
        g.add_edge("tools", "think")

        executor = g.compile()
        # Route = "finish" → should go to __end__ immediately.
        state = executor({"route": "finish"})
        assert "conditional" in state.get("visited", [])

    def test_conditional_loops_back(self):
        g = StateGraph()
        g.add_node("think", _conditional_node)
        g.add_node("tools", lambda s: {**s, "tool_executed": True, "__next__": "think"})
        g.set_entry_point("think")
        g.add_conditional_edges(
            "think",
            lambda s: s.get("__next__", "finish"),
            {"tools": "tools", "finish": "__end__"},
        )
        g.add_edge("tools", "think")

        executor = g.compile()
        # Route = "tools" → executes tools → loops back to think.
        # On the second visit to think, we force finish.
        state = {"route": "tools", "visit_count": 0}

        def _think_with_limit(s: dict) -> dict:
            count = s.get("visit_count", 0) + 1
            s["visit_count"] = count
            s.setdefault("visited", []).append("conditional")
            if count >= 2:
                s["__next__"] = "finish"
            else:
                s["__next__"] = "tools"
            return s

        g2 = StateGraph()
        g2.add_node("think", _think_with_limit)
        g2.add_node("tools", lambda s: {**s, "tool_executed": True, "__next__": "think"})
        g2.set_entry_point("think")
        g2.add_conditional_edges(
            "think",
            lambda s: s.get("__next__", "finish"),
            {"tools": "tools", "finish": "__end__"},
        )
        g2.add_edge("tools", "think")

        executor2 = g2.compile()
        state = executor2({"route": "tools", "visit_count": 0})
        assert state.get("tool_executed") is True
        assert state["visit_count"] == 2

    def test_unknown_route_key_stops(self):
        g = StateGraph()
        g.add_node("think", _conditional_node)
        g.add_node("tools", lambda s: {**s, "__next__": "think"})
        g.set_entry_point("think")
        g.add_conditional_edges(
            "think",
            lambda s: s.get("__next__", "finish"),
            {"tools": "tools", "finish": "__end__"},
            # "unknown_key" is NOT in the mapping
        )

        executor = g.compile()
        state = executor({"route": "unknown_key"})
        # Should stop because the route key isn't in the mapping.
        assert state.get("stop_reason") is None  # No crash, clean exit


class TestMaxTransitions:
    """Tests for the transition cap."""

    def test_infinite_loop_is_capped(self):
        g = StateGraph()
        g.add_node("loop", _counter_node)
        g.set_entry_point("loop")
        g.add_conditional_edges(
            "loop",
            lambda s: s.get("__next__", "finish"),
            {"counter": "loop", "finish": "__end__"},
        )

        executor = g.compile(max_transitions=2)
        state = executor({"count": 0})
        # Should stop at 2 transitions, even though counter wants to keep going.
        assert state.get("__stop_reason__") == "max_transitions (2)"

    def test_normal_completion_under_limit(self):
        g = StateGraph()
        g.add_node("loop", _counter_node)
        g.set_entry_point("loop")
        g.add_conditional_edges(
            "loop",
            lambda s: s.get("__next__", "finish"),
            {"counter": "loop", "finish": "__end__"},
        )

        executor = g.compile(max_transitions=10)
        state = executor({"count": 0})
        # Counter stops itself after 3 visits — well under the 10 limit.
        assert state["count"] == 3
        assert "__stop_reason__" not in state


class TestNodeErrors:
    """Tests for error handling in graph nodes."""

    def test_node_exception_propagates(self):
        g = StateGraph()
        g.add_node("fail", _failing_node)
        g.set_entry_point("fail")

        executor = g.compile()
        with pytest.raises(RuntimeError, match="node failure"):
            executor({})

    def test_unknown_node_in_routing(self):
        """When __next__ points to a non-existent node, the engine stops."""
        g = StateGraph()
        g.add_node("a", lambda s: {**s, "__next__": "ghost"})  # points to nonexistent
        g.set_entry_point("a")
        # No edge defined → engine reads __next__ from state.

        executor = g.compile()
        state = executor({})
        assert state.get("__stop_reason__", "").startswith("unknown_node")


class TestEdgeResolvers:
    """Tests that verify how edges are resolved."""

    def test_no_edge_uses_next_key(self):
        """When no edge is configured, __next__ from state is used."""
        g = StateGraph()
        g.add_node("a", lambda s: {**s, "__next__": "finish"})
        g.set_entry_point("a")
        # No edge at all — engine reads __next__ from state directly.
        executor = g.compile()
        state = executor({})
        # Clean finish — no stop_reason.
        assert "__stop_reason__" not in state
