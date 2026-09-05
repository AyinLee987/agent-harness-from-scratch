"""Lightweight state-graph engine for agent loops.

A :class:`StateGraph` is a directed graph where each node is a pure function
``(state) -> state`` and edges determine which node executes next. The graph
engine drives execution by repeatedly calling the current node, inspecting
the ``__next__`` key the node writes into state, and following the
corresponding edge to the next node.

Design goals:
    * Zero external dependencies (stdlib only).
    * Nodes are decoupled from routing -- the graph owns the wiring.
    * Conditional edges are rule-based (no extra LLM call for routing).
    * The engine itself is pattern-agnostic: ReAct, Plan-Execute, or
      multi-agent orchestration can all be expressed as different graphs.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterator

#: A graph node is a pure function: it receives state, does work, and returns
#: the (possibly mutated) state. The node writes a ``"__next__"`` string key
#: into the returned state dict to tell the engine which edge to follow.
NodeFn = Callable[[Dict[str, Any]], Dict[str, Any]]

#: A router function inspects state and returns the next node name (or
#: ``"__end__"`` to stop). Used by conditional edges.
RouterFn = Callable[[Dict[str, Any]], str]


class StateGraph:
    """A directed graph of named nodes connected by fixed or conditional edges.

    Usage::

        g = StateGraph()
        g.add_node("greet", greet_node)
        g.add_node("farewell", farewell_node)
        g.set_entry_point("greet")
        g.add_edge("greet", "farewell")        # fixed edge
        g.add_edge("farewell", "__end__")      # terminal
        executor = g.compile()
        state = executor({"name": "World"})
    """

    _END = "__end__"

    def __init__(self) -> None:
        self._nodes: Dict[str, NodeFn] = {}
        # _edges maps from_node -> (router_fn | fixed_target_str)
        # A fixed edge is stored as a str (the target node name).
        # A conditional edge is stored as (router_fn, mapping_dict) tuple.
        self._edges: Dict[str, str | tuple[RouterFn, Dict[str, str]]] = {}
        self._entry: str | None = None

    # -- builder API ------------------------------------------------------
    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        """Register a node function under ``name``.

        Returns ``self`` for chaining.
        """
        if not name:
            raise ValueError("Node name must be non-empty.")
        if name == self._END:
            raise ValueError(f"'{self._END}' is reserved.")
        self._nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        """Set the first node executed when the graph runs."""
        if name not in self._nodes and name != self._END:
            raise ValueError(f"Entry point '{name}' is not a registered node.")
        self._entry = name
        return self

    def add_edge(self, from_node: str, to_node: str) -> "StateGraph":
        """Add a fixed (unconditional) edge from ``from_node`` to ``to_node``.

        After ``from_node`` executes, the engine will always transition to
        ``to_node``, regardless of what the node wrote into state.
        """
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' is not registered.")
        # __end__ is always a valid target.
        self._edges[from_node] = to_node
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        router: RouterFn,
        mapping: Dict[str, str],
    ) -> "StateGraph":
        """Add conditional edges from ``from_node``.

        After ``from_node`` executes, ``router(state)`` is called. Its return
        value is looked up in ``mapping`` to determine the next node. If the
        router returns a key not in ``mapping``, execution stops.
        """
        if from_node not in self._nodes:
            raise ValueError(f"Source node '{from_node}' is not registered.")
        if not mapping:
            raise ValueError("mapping must be non-empty.")
        self._edges[from_node] = (router, mapping)
        return self

    # -- traversal --------------------------------------------------------
    def iter_steps(self, max_transitions: int = 200) -> Iterator[tuple[str, NodeFn]]:
        """Walk the graph one node at a time, letting the caller run each node.

        Yields ``(node_name, node_fn)`` and expects the state produced by
        running that node to be sent back in::

            traversal = graph.iter_steps()
            name, node = next(traversal)
            while True:
                state = my_runner(node, state)
                try:
                    name, node = traversal.send(state)
                except StopIteration:
                    break

        This is :meth:`compile` with the execution inverted. The graph still
        owns routing -- it reads ``__next__`` and follows the same edges --
        but the *caller* owns how a node gets executed. That separation is
        what lets a single graph be driven both synchronously and from an
        event loop without a second copy of the traversal (see
        ``ReActLoop._drive`` / ``_adrive`` and BUGS.md #22).

        ``compile()`` is now this plus a trivial runner, so the two cannot
        disagree about routing.
        """

        if self._entry is None:
            raise ValueError("No entry point set. Call set_entry_point() first.")

        current: str | None = self._entry
        transitions = 0
        state: Dict[str, Any] = {}

        while current is not None and current != self._END:
            if transitions >= max_transitions:
                state["__stop_reason__"] = f"max_transitions ({max_transitions})"
                break

            node_fn = self._nodes.get(current)
            if node_fn is None:
                state["__stop_reason__"] = f"unknown_node: {current}"
                break

            state = yield (current, node_fn)
            transitions += 1

            edge = self._edges.get(current)
            if edge is None:
                # No outgoing edge → node controls routing via __next__.
                next_key: str = state.pop("__next__", self._END)
                if next_key in (self._END, "finish"):
                    break
                if next_key not in self._nodes:
                    state["__stop_reason__"] = f"unknown_node: {next_key}"
                    break
                current = next_key
            elif isinstance(edge, str):
                # Fixed edge → always go to the target.
                current = edge
            else:
                # Conditional edge → call router, look up mapping.
                router, mapping = edge
                current = mapping.get(router(state), self._END)

    # -- compile ----------------------------------------------------------
    def compile(self, max_transitions: int = 200) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        """Return a runnable executor function.

        Args:
            max_transitions: Hard cap on node-to-node hops (prevents infinite
                loops in misconfigured graphs).

        Returns:
            A callable ``(state) -> state`` that drives the graph loop.

        Implemented over :meth:`iter_steps` with the simplest possible
        runner -- call the node, take what it returns -- so there is exactly
        one copy of the routing logic.
        """

        # Checked here as well as in iter_steps: a generator's body does not
        # run until it is first advanced, so relying on iter_steps' own check
        # would defer a misconfigured graph's failure from compile() to the
        # first request, which is the wrong place to find out.
        if self._entry is None:
            raise ValueError("No entry point set. Call set_entry_point() first.")

        def executor(state: Dict[str, Any]) -> Dict[str, Any]:
            traversal = self.iter_steps(max_transitions)
            try:
                _name, node_fn = next(traversal)
            except StopIteration:
                return state
            while True:
                state = node_fn(state)
                try:
                    _name, node_fn = traversal.send(state)
                except StopIteration:
                    return state

        return executor
