"""Verbose-description accuracy test: same 50 tools, ~6.5x longer descriptions.

Neither raw tool count (``tool_scaling_test.py``, 50 tools, 100% accuracy)
nor call-chain length (``tool_scaling_multi_test.py``, 14 chained tasks,
100% accuracy) moved the needle for DeepSeek-chat. This script isolates the
other half of the original interview framing -- "too many tools -> tool
*descriptions* get long -> that's what actually hurts accuracy" -- by
holding tool count fixed at 50 and swapping in
``tool_scaling_verbose_kit.ALL_TOOLS_VERBOSE``, whose descriptions carry
~6.5x more (deliberately repetitive boilerplate) text than the concise kit,
while every name/parameter/behavior stays identical.

Runs both:
  1. The same 6 single-tool probe tasks used in tool_scaling_test.py.
  2. All 14 chained tasks from tool_scaling_multi_test.py.

...against the verbose 50-tool registry, so results are directly comparable
to the concise-kit numbers already recorded in the README.

Usage
-----
    python examples/tool_scaling_verbose_test.py
    python examples/tool_scaling_verbose_test.py --provider bailian
    python examples/tool_scaling_verbose_test.py --dump verbose_results.json
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
from tool_scaling_verbose_kit import ALL_TOOLS_VERBOSE  # noqa: E402

SINGLE_TASKS_PATH = os.path.join(_HERE, "tool_scaling_tasks.json")
MULTI_TASKS_PATH = os.path.join(_HERE, "tool_scaling_multi_tasks.json")

# Same probe set as tool_scaling_test.py, for apples-to-apples comparison.
PROBE_TASK_IDS = [
    "math_modulo",
    "text_replace",
    "date_days_between",
    "convert_kg_to_lb",
    "data_json_validate",
    "text_is_palindrome",
]


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


def _load(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_single(provider: str, registry: ToolRegistry, delay: float) -> dict:
    tasks_by_id = {t["id"]: t for t in _load(SINGLE_TASKS_PATH)}
    tasks = [tasks_by_id[tid] for tid in PROBE_TASK_IDS]
    correct = 0
    rows = []
    for task in tasks:
        llm = _make_llm(provider)
        agent = ReActAgent(llm=llm, tools=registry, max_steps=5)
        outcome = agent.run(task["prompt"])
        used = any((s.get("action") or {}).get("name") == task["expect_tool"] for s in outcome.trajectory)
        correct += int(used)
        rows.append({"task_id": task["id"], "used_expected_tool": used})
        if delay:
            time.sleep(delay)
    return {"kind": "single_probe", "total": len(tasks), "correct": correct,
            "accuracy": correct / len(tasks), "rows": rows}


def run_multi(provider: str, registry: ToolRegistry, max_steps: int, delay: float) -> dict:
    tasks = _load(MULTI_TASKS_PATH)
    exact = 0
    rows = []
    for task in tasks:
        llm = _make_llm(provider)
        agent = ReActAgent(llm=llm, tools=registry, max_steps=max_steps)
        try:
            outcome = agent.run(task["prompt"])
        except Exception as exc:  # noqa: BLE001
            rows.append({"task_id": task["id"], "error": str(exc), "exact_sequence_match": False})
            if delay:
                time.sleep(delay)
            continue
        called = [(s.get("action") or {}).get("name") for s in outcome.trajectory if s.get("action")]
        match = called == task["expect_tool_sequence"]
        exact += int(match)
        rows.append({"task_id": task["id"], "expect_tool_sequence": task["expect_tool_sequence"],
                     "called_sequence": called, "exact_sequence_match": match})
        if delay:
            time.sleep(delay)
    return {"kind": "multi_chain", "total": len(tasks), "correct": exact,
            "accuracy": exact / len(tasks), "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verbose-description tool-calling accuracy test.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "bailian", "openai"])
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--dump", metavar="PATH")
    args = parser.parse_args()

    concise_chars = len(json.dumps([t.to_schema() for t in ALL_TOOLS]))
    verbose_chars = len(json.dumps([t.to_schema() for t in ALL_TOOLS_VERBOSE]))
    registry = ToolRegistry(ALL_TOOLS_VERBOSE)

    print("=" * 74)
    print(f"VERBOSE-DESCRIPTION ACCURACY TEST  (provider={args.provider}, 50 tools)")
    print(f"Schema size: concise={concise_chars} chars vs verbose={verbose_chars} chars "
          f"({verbose_chars / concise_chars:.1f}x)")
    print("=" * 74)

    single = run_single(args.provider, registry, args.delay)
    print(f"Single-tool probe accuracy (verbose descriptions): "
          f"{single['accuracy']:.0%} ({single['correct']}/{single['total']})  "
          f"[concise-kit baseline: 100%]")

    multi = run_multi(args.provider, registry, args.max_steps, args.delay)
    print(f"Multi-tool chain accuracy  (verbose descriptions): "
          f"{multi['accuracy']:.0%} ({multi['correct']}/{multi['total']})  "
          f"[concise-kit baseline: 100%]")
    print("=" * 74)

    report = {
        "provider": args.provider,
        "concise_schema_chars": concise_chars,
        "verbose_schema_chars": verbose_chars,
        "growth_factor": verbose_chars / concise_chars,
        "single_probe": single,
        "multi_chain": multi,
    }
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()
