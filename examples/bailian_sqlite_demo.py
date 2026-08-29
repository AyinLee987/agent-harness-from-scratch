"""Demo: Persistent SQLite vector memory + Bailian embeddings.

Run it directly::

    # Without API key — uses MockLLM (deterministic, zero cost):
    python examples/bailian_sqlite_demo.py

    # With 百炼 (Bailian) API key — real embeddings:
    set BAILIAN_API_KEY=sk-...
    python examples/bailian_sqlite_demo.py

What this demonstrates:
    1. SQLiteVectorStore persists embeddings to disk (survives process restart).
    2. The application explicitly chooses which successful outcomes to remember.
    3. Vector recall: semantic search returns relevant past memories.
    4. Run it twice — second run shows memories loaded from disk.

Memory persistence flow::

    agent.run("What is 23 times 17?")
      │
      ├─ think → act → observe (normal ReAct loop)
      ├─ answer: "391"
      └─ application policy approves an explicit long_term.add(...)
              │
              └─ SQLiteVectorStore.add()
                      │
                      └─ INSERT INTO vectors (id, text, embedding, ...)
                              │
                              └─ persisted to memory/demo_store.db
"""

from __future__ import annotations

import ast
import operator
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow running this file directly from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import (
    BailianLLM,
    LongTermMemory,
    MockLLM,
    OpenAILLM,
    ReActAgent,
    SQLiteVectorStore,
    ToolRegistry,
    tool,
)

# ------- Safe calculator tool -----------------------------------------------

_ALLOWED_OPS: dict[type, Any] = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
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
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"
    if result == int(result):
        return str(int(result))
    return str(result)


@tool
def datetime_now() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


# ------- Agent builder -------------------------------------------------------


def build_agent(llm, db_path: str | None = None) -> ReActAgent:
    """Build a ReAct agent with persistent SQLite-backed memory."""

    # Rename to "datetime" so MockLLM's heuristics pick it up.
    datetime_now.name = "datetime"
    tools = ToolRegistry([calculator, datetime_now])

    # Persistent vector store — survives process restarts.
    store = SQLiteVectorStore(db_path)
    long_term = LongTermMemory(llm, vector_store=store)

    return ReActAgent(
        llm=llm,
        tools=tools,
        long_term=long_term,
        max_steps=6,
    )


# ------- LLM selection -------------------------------------------------------


def _pick_llm():
    """Select the best available LLM provider.

    Priority: BAILIAN_API_KEY → OPENAI_API_KEY → MockLLM (free, offline).
    """
    if os.environ.get("BAILIAN_API_KEY"):
        print("[llm] Using Bailian (百炼) — qwen-plus + text-embedding-v3")
        return BailianLLM()

    if os.environ.get("OPENAI_API_KEY"):
        print("[llm] Using OpenAI")
        return OpenAILLM()

    print("[llm] Using MockLLM (no API key set — deterministic, offline)")
    return MockLLM()


# ------- Main ----------------------------------------------------------------


DB_PATH = str(Path(__file__).resolve().parent.parent / "memory" / "demo_store.db")


def main() -> None:
    llm = _pick_llm()
    agent = build_agent(llm, db_path=DB_PATH)

    # ── Step 1: Check if memories already exist on disk ───────────────────
    existing = agent.long_term.list_all()
    if existing:
        print(f"\n[memory] Found {len(existing)} existing record(s) from a previous run:")
        for rec in existing:
            print(f"  [{rec.id}] {rec.text[:120]}")

    # ── Step 2: Run queries; explicitly persist approved demo outcomes ──
    queries = [
        "What is 23 times 17?",
        "What is the result of 144 divided by 12?",
        "What is 500 minus 123?",
        "What is today's date?",
    ]

    for q in queries:
        print(f"\n>>> {q}")
        result = agent.run(q)
        print(f"    ANSWER: {result.answer}")
        print(f"    (steps={result.steps}, tokens={result.tokens}, "
              f"stop={result.stop_reason})")
        if result.success:
            # Demo-only allow-list policy. ReActAgent itself never auto-persists
            # model answers; production code should use MemoryManager policies.
            agent.long_term.add(
                f"Verified demo outcome: {q} -> {result.answer}",
                {"source_type": "explicit_demo_policy"},
            )

    # ── Step 3: Show accumulated knowledge ───────────────────────────────
    print(f"\n[memory] Total records in SQLite: {len(agent.long_term)}")
    for rec in agent.long_term.list_all():
        print(f"  [{rec.id}] {rec.text[:120]}")

    # ── Step 4: Semantic recall — find relevant past memories ────────────
    print('\n>>> Recall test: "math with numbers"')
    results = agent.long_term.search("math with numbers", k=3)
    for text, score in results:
        print(f"    [score={score:.4f}] {text[:120]}")

    # ── Step 5: Delete a record ──────────────────────────────────────────
    if existing or len(agent.long_term) > 0:
        all_records = agent.long_term.list_all()
        to_delete = all_records[0]
        print(f"\n[db] Deleting record: {to_delete.id}")
        agent.long_term.delete(to_delete.id)
        print(f"[db] Remaining: {len(agent.long_term)} record(s)")

    print(f"\n[db] SQLite file: {DB_PATH}")
    print("[tip] Run this script again — memories from SQLite persist!")


if __name__ == "__main__":
    main()
