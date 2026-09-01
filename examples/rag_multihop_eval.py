"""Does query decomposition (agent/rag/decomposition.py) actually help
parallel-composite and sequential-dependent ("multi-hop") questions?

Runs every task in ``rag_multihop_tasks.json`` twice through a full
Leader-shaped agent (mandatory RAGContextProvider injection +
medical_evidence_search follow-up tool, same wiring app/server.py uses)
against the synthetic corpus in ``rag_multihop_kit.py``:

    baseline  -- no decomposer, today's plain behavior.
    treatment -- LLMQueryDecomposer wired into the pipeline.

Scores whether the final answer actually contains every expected fact
(``expect_substrings``), and how many ``medical_evidence_search`` follow-up
calls the model made -- the treatment's whole value proposition for
"sequential" tasks is a hint nudging the model to make *more* of these.

Usage
-----
    python examples/rag_multihop_eval.py
    python examples/rag_multihop_eval.py --provider bailian
    python examples/rag_multihop_eval.py --dump multihop_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> local imports

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent import LLMQueryDecomposer, OpenAICompatibleEmbeddingProvider  # noqa: E402

from rag_multihop_kit import build_agent, build_pipeline  # noqa: E402

TASKS_PATH = os.path.join(_HERE, "rag_multihop_tasks.json")


def _make_llm(provider: str):
    if provider == "deepseek":
        from agent import DeepSeekLLM
        return DeepSeekLLM()
    if provider == "bailian":
        from agent import BailianLLM
        return BailianLLM()
    if provider == "openai":
        from agent import OpenAILLM
        return OpenAILLM()
    raise ValueError(f"Unknown provider: {provider}")


def _make_embeddings():
    model = os.getenv("RAG_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBED_MODEL")
    if not model:
        raise RuntimeError(
            "Set RAG_EMBEDDING_MODEL (+ RAG_EMBEDDING_API_KEY/RAG_EMBEDDING_BASE_URL) in .env."
        )
    api_key = os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    return OpenAICompatibleEmbeddingProvider(
        model=model, api_key=api_key, base_url=base_url, provider_name="multihop-eval",
    )


def _load_tasks() -> List[Dict[str, Any]]:
    with open(TASKS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _run_condition(
    provider: str, use_decomposer: bool, tasks: List[Dict[str, Any]], delay: float,
) -> List[Dict[str, Any]]:
    rows = []
    for task in tasks:
        decomposer = LLMQueryDecomposer(_make_llm(provider)) if use_decomposer else None
        pipeline = build_pipeline(_make_embeddings(), decomposer=decomposer)
        agent = build_agent(lambda: _make_llm(provider), pipeline)
        try:
            outcome = agent.run(task["prompt"])
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            rows.append({
                "task_id": task["id"], "type": task["type"], "error": str(exc),
                "answer_ok": False, "search_calls": 0,
            })
            print(f"  [{task['type']:<10}] {task['id']:<30} ERROR: {exc}")
            if delay:
                time.sleep(delay)
            continue

        called = [
            (step.get("action") or {}).get("name")
            for step in outcome.trajectory if step.get("action")
        ]
        search_calls = called.count("medical_evidence_search")
        answer_ok = all(s.lower() in outcome.answer.lower() for s in task["expect_substrings"])
        print(f"  [{task['type']:<10}] {task['id']:<30} answer={'ok' if answer_ok else 'x':<3} "
              f"search_calls={search_calls}")
        rows.append({
            "task_id": task["id"], "type": task["type"], "answer_ok": answer_ok,
            "search_calls": search_calls, "answer": outcome.answer,
            "stop_reason": outcome.stop_reason,
        })
        if delay:
            time.sleep(delay)
    return rows


def _summarize(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    by_type: Dict[str, List[bool]] = {}
    for row in rows:
        by_type.setdefault(row["type"], []).append(row["answer_ok"])
    print(f"\n{label}:")
    for type_name, oks in sorted(by_type.items()):
        print(f"  {type_name:<12} {sum(oks)}/{len(oks)} correct")
    total_ok = sum(row["answer_ok"] for row in rows)
    avg_calls = sum(row["search_calls"] for row in rows) / len(rows) if rows else 0.0
    print(f"  overall: {total_ok}/{len(rows)}   avg medical_evidence_search calls/task: {avg_calls:.2f}")
    return {
        "by_type": {t: f"{sum(oks)}/{len(oks)}" for t, oks in by_type.items()},
        "overall": f"{total_ok}/{len(rows)}",
        "avg_search_calls": avg_calls,
    }


def run(provider: str, delay: float) -> Dict[str, Any]:
    tasks = _load_tasks()
    print("=" * 78)
    print(f"RAG MULTI-HOP: does query decomposition help?  (provider={provider})")
    print(f"Tasks: {len(tasks)}  types: {sorted({t['type'] for t in tasks})}")
    print("=" * 78)

    print("\n--- BASELINE (no decomposer) ---")
    baseline = _run_condition(provider, False, tasks, delay)
    print("\n--- TREATMENT (LLMQueryDecomposer wired in) ---")
    treatment = _run_condition(provider, True, tasks, delay)

    print("\n" + "-" * 78)
    baseline_summary = _summarize(baseline, "BASELINE")
    treatment_summary = _summarize(treatment, "TREATMENT")
    print("=" * 78)

    return {
        "provider": provider,
        "baseline": {"rows": baseline, "summary": baseline_summary},
        "treatment": {"rows": treatment, "summary": treatment_summary},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG multi-hop / query-decomposition eval.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "bailian", "openai"])
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--dump", metavar="PATH")
    args = parser.parse_args()

    report = run(args.provider, args.delay)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()
