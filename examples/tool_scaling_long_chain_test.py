"""Long tool-call chain accuracy test: does chain *length* itself matter?

``tool_scaling_multi_test.py`` originally tested only 2-step chains (14/14
exact matches at 50 tools, 13/14 at 100 -- one task picked up an extra,
harmless self-check call); its tasks were since extended to 5+ steps too,
so both task files now probe the same "5+ step chain" regime. This script
asks the length question directly: does accuracy degrade further as the
*chain itself* gets longer, independent of registry size?

Loads ``tool_scaling_long_chain_tasks.json`` -- 7 tasks, every
``expect_tool_sequence`` at least 5 steps long -- and runs them against the
full tool registry from ``tool_scaling_kit.py`` (currently 100 tools),
scoring exact tool-sequence match and final-answer correctness, broken down
by chain length.

Usage
-----
    python examples/tool_scaling_long_chain_test.py
    python examples/tool_scaling_long_chain_test.py --provider bailian
    python examples/tool_scaling_long_chain_test.py --dump long_chain_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> local imports

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent import ReActAgent, ToolRegistry  # noqa: E402

from tool_scaling_kit import ALL_TOOLS  # noqa: E402

TASKS_PATH = os.path.join(_HERE, "tool_scaling_long_chain_tasks.json")


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


def _load_tasks() -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run(provider: str, max_steps: int, delay: float) -> dict:
    tasks = _load_tasks()
    registry = ToolRegistry(ALL_TOOLS)
    tool_names = {t.name for t in ALL_TOOLS}
    for task in tasks:
        unknown = [t for t in task["expect_tool_sequence"] if t not in tool_names]
        if unknown:
            raise RuntimeError(f"Task {task['id']} references unknown tools: {unknown}")

    print("=" * 78)
    print(f"LONG TOOL-CHAIN ACCURACY  (provider={provider}, registry size={len(ALL_TOOLS)})")
    print(f"Tasks: {len(tasks)}  (chain lengths: "
          f"{sorted(len(t['expect_tool_sequence']) for t in tasks)})")
    print("=" * 78)

    rows = []
    by_length: dict[int, list[bool]] = defaultdict(list)
    for task in tasks:
        expected = task["expect_tool_sequence"]
        length = len(expected)
        llm = _make_llm(provider)
        agent = ReActAgent(llm=llm, tools=registry, max_steps=max_steps)
        try:
            outcome = agent.run(task["prompt"])
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            rows.append({"task_id": task["id"], "chain_length": length, "error": str(exc),
                         "exact_sequence_match": False, "answer_ok": False})
            by_length[length].append(False)
            print(f"  [{length}] {task['id']:<28} ERROR: {exc}")
            if delay:
                time.sleep(delay)
            continue

        called = [
            (step.get("action") or {}).get("name")
            for step in outcome.trajectory if step.get("action")
        ]
        exact = called == expected
        answer_ok = all(s.lower() in outcome.answer.lower() for s in task.get("expect_substrings", []))
        by_length[length].append(exact)

        print(f"  [{length}] {task['id']:<28} seq={'ok' if exact else 'x':<3} "
              f"answer={'ok' if answer_ok else 'x':<3} called_len={len(called)}")
        if not exact:
            print(f"        expected={expected}")
            print(f"        called  ={called}")

        rows.append({
            "task_id": task["id"],
            "chain_length": length,
            "expect_tool_sequence": expected,
            "called_sequence": called,
            "exact_sequence_match": exact,
            "answer_ok": answer_ok,
            "answer": outcome.answer,
            "steps": outcome.steps,
            "stop_reason": outcome.stop_reason,
        })
        if delay:
            time.sleep(delay)

    total = len(tasks)
    exact_total = sum(1 for r in rows if r.get("exact_sequence_match"))
    answer_total = sum(1 for r in rows if r.get("answer_ok"))

    print("-" * 78)
    for length in sorted(by_length):
        results = by_length[length]
        acc = sum(results) / len(results)
        print(f"  chain length {length}: {acc:.0%} exact-match ({sum(results)}/{len(results)})")
    print(f"Overall exact-sequence accuracy: {exact_total / total:.0%} ({exact_total}/{total})")
    print(f"Overall final-answer accuracy:   {answer_total / total:.0%} ({answer_total}/{total})")
    print("=" * 78)

    return {
        "provider": provider,
        "registry_size": len(ALL_TOOLS),
        "total": total,
        "exact_sequence_accuracy": exact_total / total,
        "answer_accuracy": answer_total / total,
        "by_chain_length": {
            str(length): sum(results) / len(results) for length, results in by_length.items()
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Long tool-call-chain accuracy test.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "bailian", "openai"])
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--dump", metavar="PATH")
    args = parser.parse_args()

    report = run(args.provider, args.max_steps, args.delay)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()
