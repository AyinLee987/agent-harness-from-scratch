"""Example: a ReAct agent with calculator, web-search-stub, datetime, and
memory_search (RAG-as-a-tool) capabilities.

Run it directly to watch the agent reason and call tools::

    python examples/basic_tools.py

Key design pattern demonstrated:
    Long-term memory retrieval (RAG) is registered as a **tool** so the model
    decides *when* to search — no auto-injection into the system prompt. This
    saves tokens when memory is irrelevant and gives the model agency over
    what to search.

By default it uses the dependency-free :class:`MockLLM`, so it works without an
API key. Set ``OPENAI_API_KEY`` (and ``USE_OPENAI=1``) to drive a real model.
"""

from __future__ import annotations

import ast
import operator
import os
import sys
from datetime import datetime, timezone
from typing import Any

# Allow running this file directly from the repo root (`python examples/...`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import LongTermMemory, MockLLM, OpenAILLM, ReActAgent, ToolRegistry, tool

# A small, deterministic "knowledge base" backing the web-search stub. Keeping
# results canned makes demos and eval runs reproducible (and keeps the repo
# strictly clean-room -- generic public facts only).
_SEARCH_KB = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "tallest mountain": "Mount Everest is the tallest mountain on Earth at 8,849 m.",
    "speed of light": "The speed of light is approximately 299,792 km/s.",
    "creator of python": "Python was created by Guido van Rossum.",
    "python": "Python was created by Guido van Rossum.",
}

# Operators allowed by the safe expression evaluator.
_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an arithmetic AST without using ``eval``."""

    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric constants are allowed.")
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.

    Args:
        expression: An arithmetic expression, e.g. '23 * 17' or '(12 + 8) * 5'.
    """

    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as exc:  # noqa: BLE001 - report the error back to the agent
        return f"Could not evaluate '{expression}': {exc}"
    # Render whole numbers without a trailing .0 so substring checks are natural.
    if result == int(result):
        return str(int(result))
    return str(result)


@tool
def web_search(query: str) -> str:
    """Look up a fact from a small canned knowledge base (offline stub).

    Args:
        query: The search query.
    """

    q = query.lower()
    for key, value in _SEARCH_KB.items():
        if key in q:
            return value
    return f"No results found for '{query}'."


@tool
def datetime_now() -> str:
    """Return the current UTC date and time in ISO-8601 format."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_registry() -> ToolRegistry:
    """Build the demo tool registry.

    The decorator names tools after their functions; we rename ``datetime_now``
    to ``datetime`` to match the names the MockLLM's heuristics look for.
    """

    registry = ToolRegistry([calculator, web_search])
    datetime_now.name = "datetime"
    registry.register(datetime_now)
    return registry


def build_agent() -> ReActAgent:
    """Construct a ReAct agent wired with the demo tools and an LLM."""

    if os.environ.get("USE_OPENAI") and os.environ.get("OPENAI_API_KEY"):
        llm = OpenAILLM()
    else:
        llm = MockLLM()
    return ReActAgent(llm=llm, tools=build_registry(), max_steps=6)


# ---------------------------------------------------------------------------
# Tool-based RAG: memory_search registered as a regular tool.
# The model decides whether and when to search long-term memory.
# ---------------------------------------------------------------------------

def _build_knowledge_memory(llm: MockLLM | OpenAILLM) -> LongTermMemory:
    """Pre-populate a :class:`LongTermMemory` with domain facts.

    These facts are *not* auto-injected — they are only retrieved when the
    model explicitly calls the ``memory_search`` tool.
    """

    mem = LongTermMemory(llm)
    facts = [
        "The company was founded in 2019 and is headquartered in Beijing.",
        "Product A (Pro Edition) costs $299 per user per year.",
        "Product B (Enterprise Edition) costs $599 per user per year.",
        "The refund policy allows returns within 30 days of purchase.",
        "Customer support is available 24/7 via email at support@example.com.",
        "The CEO is Zhang Wei, previously CTO at a Fortune 500 company.",
        "Office locations: Beijing (HQ), Shanghai, Shenzhen, and Singapore.",
        "The company has served over 50,000 enterprise customers since launch.",
    ]
    for fact in facts:
        mem.add(fact, {"source": "company_kb"})
    return mem


def build_registry_with_memory(llm: MockLLM | OpenAILLM) -> ToolRegistry:
    """Same as :func:`build_registry`, but includes ``memory_search``."""

    registry = build_registry()
    mem = _build_knowledge_memory(llm)
    registry.register(mem.as_search_tool())
    return registry


def build_agent_with_memory() -> ReActAgent:
    """Construct a ReAct agent with memory_search as a tool.

    The long-term memory is pre-populated with domain knowledge, but retrieval
    is tool-driven: the model decides *when* to search.
    """

    if os.environ.get("USE_OPENAI") and os.environ.get("OPENAI_API_KEY"):
        llm = OpenAILLM()
    else:
        llm = MockLLM()
    tools = build_registry_with_memory(llm)
    return ReActAgent(llm=llm, tools=tools, max_steps=8)


def main() -> None:
    print("=" * 60)
    print("PART 1: Basic tools (calculator, web_search, datetime)")
    print("=" * 60)

    agent = build_agent()
    demos = [
        "What is 23 times 17?",
        "Search for the capital of France.",
        "What is today's date?",
    ]
    for task in demos:
        print(f"\n>>> TASK: {task}")
        result = agent.run(task)
        for step in result.trajectory:
            if step["action"]:
                print(f"    [step {step['index']}] action={step['action']['name']} "
                      f"args={step['action']['arguments']} -> {step['observation']}")
        print(f"    ANSWER: {result.answer}")
        print(f"    (steps={result.steps}, tokens={result.tokens}, "
              f"stop={result.stop_reason})")

    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 2: Tool-based RAG — memory_search as a tool")
    print("=" * 60)
    print(
        "The agent has 8 domain facts pre-loaded in LongTermMemory.\n"
        "It decides *itself* whether to call memory_search — "
        "no auto-injection."
    )

    mem_agent = build_agent_with_memory()
    rag_demos = [
        "What is the refund policy?",
        "How much does Product A cost?",
        "What is 42 plus 58?",
    ]
    for task in rag_demos:
        print(f"\n>>> TASK: {task}")
        result = mem_agent.run(task)
        for step in result.trajectory:
            if step["action"]:
                print(f"    [step {step['index']}] action={step['action']['name']} "
                      f"args={step['action']['arguments']} -> {step['observation']}")
        print(f"    ANSWER: {result.answer}")
        print(f"    (steps={result.steps}, tokens={result.tokens}, "
              f"stop={result.stop_reason})")


if __name__ == "__main__":
    main()
