"""Does the hierarchical (main + specialist-subagent) design raise accuracy?

Reruns the exact same 21 five-step-chain tasks that
``tool_scaling_multi_test.py`` (14) and ``tool_scaling_long_chain_test.py``
(7) ran against the *flat* 100-tool registry -- where exact-sequence
accuracy landed at 57% in both files -- through ``hierarchical_agent_kit``'s
main-agent-plus-specialist-subagents setup instead.

Scoring, kept apples-to-apples with the flat baseline:
    exact_sequence_match -- flatten every specialist call's own tool-call
                             sequence, in delegation order, and compare
                             against the task's expect_tool_sequence exactly
                             (same metric the flat experiments used).
    answer_ok             -- the main agent's final answer contains
                              expect_substrings.
    delegate_calls         -- how many specialist delegations the task took
                              (1 if every step happened to fall in one
                              group, more if the chain crosses groups).
    any_task_failed         -- whether any specialist reported
                              TASK_FAILED (tools it had weren't enough).

Usage
-----
    python examples/hierarchical_agent_test.py
    python examples/hierarchical_agent_test.py --provider bailian
    python examples/hierarchical_agent_test.py --dump hierarchical_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> local imports

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent import ReActAgent  # noqa: E402

from hierarchical_agent_kit import MAIN_SYSTEM_PROMPT, build_main_registry  # noqa: E402

MULTI_TASKS_PATH = os.path.join(_HERE, "tool_scaling_multi_tasks.json")
LONG_CHAIN_TASKS_PATH = os.path.join(_HERE, "tool_scaling_long_chain_tasks.json")


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


def _load_tasks() -> List[Dict[str, Any]]:
    tasks = []
    with open(MULTI_TASKS_PATH, "r", encoding="utf-8") as fh:
        for t in json.load(fh):
            tasks.append({**t, "source": "multi"})
    with open(LONG_CHAIN_TASKS_PATH, "r", encoding="utf-8") as fh:
        for t in json.load(fh):
            tasks.append({**t, "source": "long_chain"})
    return tasks


def run(provider: str, max_steps: int, sub_max_steps: int, delay: float) -> Dict[str, Any]:
    tasks = _load_tasks()

    print("=" * 78)
    print(f"HIERARCHICAL (main + specialist-subagent) ACCURACY  (provider={provider})")
    print(f"Tasks: {len(tasks)}  (same 21 five-step tasks as the flat-registry runs)")
    print("=" * 78)

    rows = []
    exact_matches = 0
    answer_oks = 0
    for task in tasks:
        call_log: List[Dict[str, Any]] = []
        registry = build_main_registry(
            llm_factory=lambda: _make_llm(provider),
            call_log=call_log,
            max_steps=sub_max_steps,
        )
        main_llm = _make_llm(provider)
        main_agent = ReActAgent(
            llm=main_llm,
            tools=registry,
            system_prompt=MAIN_SYSTEM_PROMPT,
            max_steps=max_steps,
        )
        try:
            outcome = main_agent.run(task["prompt"])
        except Exception as exc:  # noqa: BLE001 - keep the sweep going
            rows.append({
                "task_id": task["id"], "source": task["source"], "error": str(exc),
                "exact_sequence_match": False, "answer_ok": False,
                "delegate_calls": len(call_log), "call_log": call_log,
            })
            print(f"  {task['id']:<28} ERROR: {exc}")
            if delay:
                time.sleep(delay)
            continue

        called_flat = [name for entry in call_log for name in entry["called_tools"]]
        expected = task["expect_tool_sequence"]
        exact = called_flat == expected
        answer_ok = all(s.lower() in outcome.answer.lower() for s in task.get("expect_substrings", []))
        any_failed = any("TASK_FAILED" in entry["answer"] for entry in call_log)
        exact_matches += int(exact)
        answer_oks += int(answer_ok)

        print(f"  [{task['source']:<10}] {task['id']:<28} seq={'ok' if exact else 'x':<3} "
              f"answer={'ok' if answer_ok else 'x':<3} delegate_calls={len(call_log)}"
              f"{'  TASK_FAILED seen' if any_failed else ''}")
        if not exact:
            print(f"        expected={expected}")
            print(f"        called  ={called_flat}")

        rows.append({
            "task_id": task["id"],
            "source": task["source"],
            "expect_tool_sequence": expected,
            "called_sequence": called_flat,
            "exact_sequence_match": exact,
            "answer_ok": answer_ok,
            "delegate_calls": len(call_log),
            "any_task_failed": any_failed,
            "answer": outcome.answer,
            "steps": outcome.steps,
            "stop_reason": outcome.stop_reason,
            "call_log": call_log,
        })
        if delay:
            time.sleep(delay)

    total = len(tasks)
    seq_accuracy = exact_matches / total if total else 0.0
    answer_accuracy = answer_oks / total if total else 0.0

    print("-" * 78)
    print(f"Exact tool-sequence accuracy: {seq_accuracy:.0%} ({exact_matches}/{total})")
    print(f"Final-answer accuracy:        {answer_accuracy:.0%} ({answer_oks}/{total})")
    print("Flat-registry baseline was:   57% exact-sequence / 100% final-answer (both task files)")
    print("=" * 78)

    return {
        "provider": provider,
        "total": total,
        "exact_sequence_matches": exact_matches,
        "sequence_accuracy": seq_accuracy,
        "answer_accuracy": answer_accuracy,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hierarchical main+specialist-subagent accuracy test.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "bailian", "openai"])
    parser.add_argument("--max-steps", type=int, default=14, help="Main agent's own max steps.")
    parser.add_argument("--sub-max-steps", type=int, default=8, help="Each specialist subagent's max steps.")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to sleep between tasks.")
    parser.add_argument("--dump", metavar="PATH", help="Write full results JSON to PATH.")
    args = parser.parse_args()

    report = run(args.provider, args.max_steps, args.sub_max_steps, args.delay)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()
