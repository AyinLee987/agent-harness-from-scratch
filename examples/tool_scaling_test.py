"""Empirical tool-count-vs-accuracy scaling experiment.

Motivation: "how many tools before tool-calling accuracy visibly drops?" is a
common interview/design question with no universal answer -- it depends on
the model, tool description quality, and how much the tools overlap. This
script measures it directly against this repo's own agent instead of citing
someone else's benchmark.

Method
------
A fixed set of *probe* tasks (one prompt each, spanning several tool
categories) is run repeatedly, unchanged, while the surrounding tool
registry grows from a handful of tools up to the full 50-tool kit in
``tool_scaling_kit.py``. The probe tools are always present; the *extra*
tools added at each step are pure distractors -- same shape (name +
docstring + JSON schema), never the right answer. If accuracy holds flat
across registry sizes, tool count alone isn't the bottleneck for this
model/kit; if it decays, the decay point is the number to report.

Usage
-----
    python examples/tool_scaling_test.py                  # DeepSeek by default
    python examples/tool_scaling_test.py --provider bailian
    python examples/tool_scaling_test.py --sizes 6,15,25,35,50
    python examples/tool_scaling_test.py --dump results.json

Requires a real LLM (DEEPSEEK_API_KEY or BAILIAN_API_KEY or OPENAI_API_KEY in
.env) -- MockLLM's tool selection is keyword-heuristic, not a model decision,
so it cannot exhibit (or measure) this effect.
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

TASKS_PATH = os.path.join(_HERE, "tool_scaling_tasks.json")

# Probe tasks: fixed across every registry size, spanning all five
# categories so no single category's overlap dominates the signal.
PROBE_TASK_IDS = [
    "math_modulo",
    "text_replace",
    "date_days_between",
    "convert_kg_to_lb",
    "data_json_validate",
    "text_is_palindrome",
]

DEFAULT_SIZES = [6, 15, 25, 35, 50]


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


def _load_tasks() -> dict:
    with open(TASKS_PATH, "r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    return {t["id"]: t for t in tasks}


def _build_registry(size: int, probe_tools: list, distractor_pool: list) -> ToolRegistry:
    """Probe tools are always included; distractors pad up to ``size``."""

    extra = max(0, size - len(probe_tools))
    return ToolRegistry(probe_tools + distractor_pool[:extra])


def run_experiment(provider: str, sizes: list[int], delay: float) -> dict:
    tasks_by_id = _load_tasks()
    tools_by_name = {t.name: t for t in ALL_TOOLS}

    missing = [tid for tid in PROBE_TASK_IDS if tid not in tasks_by_id]
    if missing:
        raise RuntimeError(f"Probe task ids not found in tasks file: {missing}")

    probe_tasks = [tasks_by_id[tid] for tid in PROBE_TASK_IDS]
    probe_tools = [tools_by_name[t["expect_tool"]] for t in probe_tasks]
    probe_names = {t.name for t in probe_tools}
    distractor_pool = [t for t in ALL_TOOLS if t.name not in probe_names]

    min_size = len(probe_tools)
    sizes = sorted({s for s in sizes if s >= min_size})
    if not sizes:
        raise ValueError(f"All --sizes must be >= {min_size} (number of probe tools).")

    print("=" * 70)
    print(f"TOOL-COUNT SCALING EXPERIMENT  (provider={provider})")
    print(f"Probe tasks ({len(probe_tasks)}): {', '.join(PROBE_TASK_IDS)}")
    print(f"Registry sizes tested: {sizes}")
    print("=" * 70)

    report: dict = {"provider": provider, "probe_task_ids": PROBE_TASK_IDS, "runs": []}

    for size in sizes:
        registry = _build_registry(size, probe_tools, distractor_pool)
        correct = 0
        rows = []
        for task in probe_tasks:
            llm = _make_llm(provider)  # fresh client/context per call
            agent = ReActAgent(llm=llm, tools=registry, max_steps=5)
            try:
                outcome = agent.run(task["prompt"])
            except Exception as exc:  # noqa: BLE001 - keep the sweep going
                rows.append({"task_id": task["id"], "used_expected_tool": False, "error": str(exc)})
                if delay:
                    time.sleep(delay)
                continue
            used = any(
                (step.get("action") or {}).get("name") == task["expect_tool"]
                for step in outcome.trajectory
            )
            correct += int(used)
            rows.append({
                "task_id": task["id"],
                "expect_tool": task["expect_tool"],
                "called": [
                    (step.get("action") or {}).get("name")
                    for step in outcome.trajectory if step.get("action")
                ],
                "used_expected_tool": used,
                "answer": outcome.answer,
            })
            if delay:
                time.sleep(delay)
        accuracy = correct / len(probe_tasks)
        print(f"  tools={size:>3}  accuracy={accuracy:.0%}  ({correct}/{len(probe_tasks)})")
        report["runs"].append({"tool_count": size, "accuracy": accuracy, "correct": correct,
                                "total": len(probe_tasks), "rows": rows})

    print("-" * 70)
    baseline = report["runs"][0]["accuracy"]
    worst = min(r["accuracy"] for r in report["runs"])
    if worst < baseline:
        drop_at = next(r["tool_count"] for r in report["runs"] if r["accuracy"] == worst)
        print(f"Accuracy dropped from {baseline:.0%} (tools={sizes[0]}) to "
              f"{worst:.0%} at tools={drop_at}.")
    else:
        print(f"No accuracy drop observed across {sizes[0]}-{sizes[-1]} tools for this "
              f"probe set/model -- try more tools, closer-overlapping distractors, or a "
              f"smaller model to see the effect.")
    print("=" * 70)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool-count vs tool-calling-accuracy sweep.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "bailian", "openai"])
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                         help="Comma-separated registry sizes to test, e.g. 6,15,25,35,50")
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds to sleep between API calls.")
    parser.add_argument("--dump", metavar="PATH", help="Write full results JSON to PATH.")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    report = run_experiment(args.provider, sizes, args.delay)

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()
