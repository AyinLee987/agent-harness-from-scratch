"""Multi-tool-call accuracy test: harder than single-tool selection.

``tool_scaling_test.py`` showed 100% single-tool selection accuracy for
DeepSeek-chat all the way to the full 50-tool kit. Single-tool tasks only
ask the model to pick *one* correct name out of the registry once. A chained
task asks it to do that correctly *multiple times in a row*, sometimes with
the same tool twice (e.g. add, then add again) -- errors compound across
steps, so this is a stricter probe of the same underlying question before
reaching for "just add more tools" as the next lever.

Loads ``tool_scaling_multi_tasks.json`` (14 tasks, each with an
``expect_tool_sequence`` -- the ordered, duplicates-allowed list of tool
calls a correct trajectory must contain) and runs them against the full
50-tool kit from ``tool_scaling_kit.py``.

Scoring per task:
    exact_sequence_match -- the actual tool-call names, in order, equal
                             expect_tool_sequence exactly (this is the
                             headline "multi-tool accuracy" metric).
    answer_ok             -- the final answer contains expect_substrings.

Usage
-----
    python examples/tool_scaling_multi_test.py
    python examples/tool_scaling_multi_test.py --provider bailian
    python examples/tool_scaling_multi_test.py --dump multi_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> local imports

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent import ReActAgent, ToolRegistry  # noqa: E402

from tool_scaling_kit import ALL_TOOLS  # noqa: E402

TASKS_PATH = os.path.join(_HERE, "tool_scaling_multi_tasks.json")


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
    registry = ToolRegistry(ALL_TOOLS)  # full 50-tool kit -- the current project size
    tool_names = {t.name for t in ALL_TOOLS}
    for task in tasks:
        unknown = [t for t in task["expect_tool_sequence"] if t not in tool_names]
        if unknown:
            raise RuntimeError(f"Task {task['id']} references unknown tools: {unknown}")

    print("=" * 74)
    print(f"MULTI-TOOL-CALL ACCURACY  (provider={provider}, registry size={len(ALL_TOOLS)})")
    print(f"Tasks: {len(tasks)}")
    print("=" * 74)

    rows = []
    exact_matches = 0
    answer_oks = 0
    for task in tasks:
        llm = _make_llm(provider)
        agent = ReActAgent(llm=llm, tools=registry, max_steps=max_steps)
        try:
            outcome = agent.run(task["prompt"])
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            rows.append({"task_id": task["id"], "error": str(exc), "exact_sequence_match": False, "answer_ok": False})
            print(f"  {task['id']:<26} ERROR: {exc}")
            if delay:
                time.sleep(delay)
            continue

        called = [
            (step.get("action") or {}).get("name")
            for step in outcome.trajectory if step.get("action")
        ]
        expected = task["expect_tool_sequence"]
        exact = called == expected
        answer_ok = all(s.lower() in outcome.answer.lower() for s in task.get("expect_substrings", []))
        exact_matches += int(exact)
        answer_oks += int(answer_ok)

        status = "OK" if exact else "MISMATCH"
        print(f"  {task['id']:<26} seq={'ok' if exact else 'x':<3} answer={'ok' if answer_ok else 'x':<3} "
              f"expected={expected} called={called}")

        rows.append({
            "task_id": task["id"],
            "prompt": task["prompt"],
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
    seq_accuracy = exact_matches / total if total else 0.0
    answer_accuracy = answer_oks / total if total else 0.0

    print("-" * 74)
    print(f"Exact tool-sequence accuracy: {seq_accuracy:.0%} ({exact_matches}/{total})")
    print(f"Final-answer accuracy:        {answer_accuracy:.0%} ({answer_oks}/{total})")
    print("=" * 74)

    return {
        "provider": provider,
        "registry_size": len(ALL_TOOLS),
        "total": total,
        "exact_sequence_matches": exact_matches,
        "sequence_accuracy": seq_accuracy,
        "answer_accuracy": answer_accuracy,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-tool-call (chained) accuracy test.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "bailian", "openai"])
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to sleep between API calls.")
    parser.add_argument("--dump", metavar="PATH", help="Write full results JSON to PATH.")
    args = parser.parse_args()

    report = run(args.provider, args.max_steps, args.delay)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()
