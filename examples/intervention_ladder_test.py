"""Intervention ladder: what actually fixes silent step-skipping?

The tool-scaling experiments established that tool count (6->100) and tool
description length (6.5x) cause no measurable degradation, and that the real
variable is chain length: past 5 steps the model silently skips steps it
judges redundant. This script measures a ladder of interventions against that
failure, from free to expensive.

    baseline                  the shipped DEFAULT_SYSTEM_PROMPT, verbatim --
                              which notably ends with "when you have enough
                              information ... do not call any more tools",
                              i.e. it encourages the failure
    instruction               that clause replaced by an explicit
                              completeness rule; nothing else changed
    fewshot                   instruction + two worked examples that execute
                              a redundant step
    hierarchical              main agent + 5 specialist subagents; the model
                              is told nothing, its tool surface is shrunk
    hierarchical_instruction  both, to see whether they stack

Scoring per run
---------------
    exact_sequence_match  called tool names, in order == expect_tool_sequence
    step_recall           multiset overlap / len(expected); one skipped step
                          in an 8-step chain scores 0.875, not 0
    answer_ok             final answer contains the chain's real final value
    failure_kind          skip / extra / other / match
    extra_calls           calls beyond the required multiset

skip and extra are tracked separately on purpose: an intervention that says
"never skip a step" can trade one failure for the other and look neutral on
exact match. That is exactly what `instruction` does at 8 steps.

For hierarchical conditions each specialist's own calls are flattened in
delegation order and compared against expect_tool_sequence -- the same metric
the flat conditions use, kept apples-to-apples.

Results and full methodology: examples/INTERVENTION_LADDER_RESULTS.md

Usage
-----
    python examples/intervention_ladder_test.py            # Test-A, k=1
    python examples/intervention_ladder_test.py         --tasks examples/gen_tasks_8step.json         --conditions baseline instruction fewshot hierarchical         --repeat 3 --workers 10 --max-steps 20 --dump results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> local imports

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Hundreds of runs would otherwise leave hundreds of per-run log files behind.
os.environ.setdefault("AGENT_LOG_PER_RUN", "false")

from agent import ReActAgent, ToolRegistry  # noqa: E402
from agent.trigger.react_loop import DEFAULT_SYSTEM_PROMPT  # noqa: E402

from tool_scaling_kit import ALL_TOOLS  # noqa: E402
from hierarchical_agent_kit import (  # noqa: E402
    MAIN_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPT, build_main_registry,
)

TASK_FILES = (
    os.path.join(_HERE, "tool_scaling_multi_tasks.json"),
    os.path.join(_HERE, "tool_scaling_long_chain_tasks.json"),
)


# --------------------------------------------------------------------------
# Conditions
# --------------------------------------------------------------------------

# Same as DEFAULT_SYSTEM_PROMPT but the trailing "when you have enough
# information, stop" clause -- which is the likely source of the skipping --
# is replaced with an explicit completeness rule. Everything else is kept
# byte-identical so the only variable is the step-completeness instruction.
COMPLETENESS_RULE = (
    "Execute EVERY step the user's request asks for, as a separate tool call, in the "
    "order given. Do this even when a step looks unnecessary -- for example when "
    "rounding a value that is already at the requested precision, when a conversion "
    "returns to the original unit, when the same operation is applied twice, or when "
    "you could work out the result yourself. A step you can predict the answer to "
    "still has to be executed. Only give your final answer once every requested step "
    "has actually been carried out with a tool call."
)

INSTRUCTION_PROMPT = (
    "You are a helpful ReAct agent. Reason step by step. Use the provided tools "
    "when they help answer the user's request. If a memory_search tool is available "
    "and the user asks about something you might have stored from past conversations "
    "or domain knowledge, call it proactively.\n\n"
    + COMPLETENESS_RULE
)

# The worked examples deliberately use tools that do NOT appear in any Test-A
# task (divide/round, and a meters<->feet round trip), so this demonstrates the
# *behaviour* rather than the test tasks. Caveat for the write-up: the T1
# (no-op round) and T5 (identity round trip) patterns are still shown, so gains
# on those two classes are partly in-distribution.
FEWSHOT_BLOCK = """

Two worked examples of carrying out every step.

Example 1
User: "Divide 20 by 4, round the result to 2 decimal places, then multiply by 3."
Correct behaviour: three tool calls -- math_divide(20, 4) -> 5, then
math_round(5, 2) -> 5.0 even though 5 is already exact at 2 decimal places,
then math_multiply(5.0, 3) -> 15. The rounding step is NOT skipped just
because it changes nothing.

Example 2
User: "Convert 10 meters to feet, then convert the result back to meters,
then add 5."
Correct behaviour: three tool calls -- convert_meters_to_feet(10) -> 32.808,
then convert_feet_to_meters(32.808) -> 10.0 even though this returns to the
starting value, then math_add(10.0, 5) -> 15.0. The round trip is NOT
collapsed just because the answer is predictable."""

# --- specialist prompts, for the hierarchical conditions -------------------
#
# A 15-task diagnostic (see INTERVENTION_LADDER_RESULTS.md) localised every
# skipped step in the hierarchical arm: in 4/4 cases the tool's group HAD been
# delegated to and the specialist simply did not call it. The main agent never
# under-delegated. So the specialist prompt, not MAIN_SYSTEM_PROMPT, is the
# layer any prompt intervention has to target here.
#
# That diagnostic also exposed a confound: the shipped SUBAGENT_SYSTEM_PROMPT
# ALREADY contains a completeness instruction ("call every tool call the task
# actually requires, even one whose result looks like it wouldn't change").
# The `hierarchical` arm was therefore never a pure architectural
# intervention. `hierarchical_bare` strips that sentence to separate routing
# from the prompt the kit happens to ship with.

SUBAGENT_BARE = (
    "You are {group_name}, a specialist agent. Reason step by step and use "
    "the provided tools to complete the delegated sub-task precisely. "
    "If, and only if, none of your available tools can complete the sub-task, "
    "respond with exactly: TASK_FAILED: <short reason>. Otherwise respond "
    "with a final answer and do not call any more tools."
)

# Tool-agnostic on purpose: each specialist holds a different 20-tool slice,
# so an example naming math_round would reference a tool text_agent lacks.
SUBAGENT_FEWSHOT = SUBAGENT_SYSTEM_PROMPT + (
    "\n\nWorked example of carrying out every step. Sub-task: \"take 5, round "
    "it to 2 decimal places, then multiply by 3\". Correct behaviour is three "
    "tool calls: round 5 to 2 decimals (which returns 5, unchanged), then "
    "multiply by 3. NOT two calls that skip the rounding because it changes "
    "nothing. Equally, a conversion that returns to its starting unit, or the "
    "same operation applied twice, is still executed once per request. A step "
    "whose result you can predict still has to be executed as a tool call -- "
    "and equally, do not add calls the sub-task did not ask for."
)

CONDITIONS = {
    "baseline": DEFAULT_SYSTEM_PROMPT,
    "instruction": INSTRUCTION_PROMPT,
    "fewshot": INSTRUCTION_PROMPT + FEWSHOT_BLOCK,
    # Rung 3: don't tell the model anything new, shrink what it has to look
    # at. The main agent sees 5 delegate_* tools (2,728 chars of schema)
    # instead of 100 tools (30,630); each specialist sees only its category.
    "hierarchical": MAIN_SYSTEM_PROMPT,
    "hierarchical_bare": MAIN_SYSTEM_PROMPT,
    "hierarchical_fewshot": MAIN_SYSTEM_PROMPT,
}

# Per-condition specialist prompt; None means the kit's shipped default.
SUBAGENT_PROMPTS = {
    "hierarchical": None,
    "hierarchical_bare": SUBAGENT_BARE,
    "hierarchical_fewshot": SUBAGENT_FEWSHOT,
}

# Conditions that run through the main-agent + specialist-subagent kit rather
# than a flat 100-tool registry.
HIERARCHICAL = set(SUBAGENT_PROMPTS)


# --------------------------------------------------------------------------
# Trigger classification (PLAN.md 5.3)
# --------------------------------------------------------------------------

_INFERABLE_BOOL = {
    "calendar_is_weekend", "date_is_leap_year", "text_is_palindrome",
    "encode_is_ascii", "calendar_is_same_day", "data_json_validate",
}
_SMALL_STAT = {
    "stat_sum", "stat_count", "math_average", "stat_product", "stat_median",
    "stat_mode", "stat_range",
}
_INVERSE_PAIRS = [
    ("convert_celsius_to_fahrenheit", "convert_fahrenheit_to_celsius"),
    ("convert_km_to_miles", "convert_miles_to_km"),
    ("convert_kg_to_lbs", "convert_lbs_to_kg"),
    ("convert_meters_to_feet", "convert_feet_to_meters"),
    ("convert_liters_to_gallons", "convert_gallons_to_liters"),
    ("data_base64_encode", "data_base64_decode"),
    ("data_url_encode", "data_url_decode"),
    ("encode_hex_encode", "encode_hex_decode"),
    ("encode_caesar_cipher", "encode_caesar_decipher"),
    ("measure_miles_to_nautical_miles", "measure_nautical_miles_to_miles"),
    ("measure_acres_to_hectares", "measure_hectares_to_acres"),
    ("measure_sqft_to_sqm", "measure_sqm_to_sqft"),
    ("measure_mph_to_kmh", "measure_kmh_to_mph"),
    ("measure_bytes_to_megabytes", "measure_megabytes_to_bytes"),
    ("text_uppercase", "text_lowercase"),
    ("date_add_days", "calendar_subtract_days"),
]


def classify_triggers(expected: list[str]) -> list[str]:
    """Tag a task with the redundancy pressures its chain contains."""
    seq = set(expected)
    tags = []
    if "math_round" in seq:
        tags.append("T1_noop_round")
    if seq & _INFERABLE_BOOL:
        tags.append("T2_inferable_bool")
    if seq & _SMALL_STAT:
        tags.append("T3_small_stat")
    if any(c > 1 for c in Counter(expected).values()):
        tags.append("T4_same_tool_twice")
    if any(a in seq and b in seq for a, b in _INVERSE_PAIRS):
        tags.append("T5_identity_roundtrip")
    return tags or ["T0_none"]


def classify_failure(called: list[str], expected: list[str]) -> str:
    """skip / extra / other -- the Stage-2 rollout bucket taxonomy."""
    if called == expected:
        return "match"

    # Is `called` an ordered subsequence of `expected` (steps dropped, nothing
    # invented)? That is the failure this project is about.
    it = iter(expected)
    if all(name in it for name in called):
        return "skip"

    # Is `expected` an ordered subsequence of `called` (everything done, plus
    # unrequested calls)? That is over-correction.
    it = iter(called)
    if all(name in it for name in expected):
        return "extra"

    return "other"


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

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


def _load_tasks(paths: tuple[str, ...] = TASK_FILES) -> list[dict]:
    tasks: list[dict] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            for task in json.load(fh):
                if task["id"] in seen:
                    raise RuntimeError(f"Duplicate task id across files: {task['id']}")
                seen.add(task["id"])
                task["_source"] = os.path.basename(path)
                # Generated tasks carry their trigger classes in _meta; hand
                # written ones get them derived from the expected sequence.
                task["_triggers"] = (
                    task.get("_meta", {}).get("trigger_classes")
                    or classify_triggers(task["expect_tool_sequence"])
                )
                tasks.append(task)
    return tasks


def step_recall(called: list[str], expected: list[str]) -> tuple[float, int]:
    """Fraction of required calls actually made, and how many extra were made.

    Multiset-based, so a chain needing math_add twice only gets full credit if
    it was called twice. Much lower variance than all-or-nothing exact match:
    one skipped step in an 8-step chain scores 0.875 rather than 0.
    """
    made = Counter(called) & Counter(expected)
    hit = sum(made.values())
    return (hit / len(expected) if expected else 0.0), len(called) - hit


_PRINT_LOCK = threading.Lock()


def _one_run(task: dict, trial: int, system_prompt: str, provider: str,
             max_steps: int, delay: float, hierarchical: bool = False,
             sub_max_steps: int = 8, subagent_prompt: str | None = None) -> dict:
    expected = task["expect_tool_sequence"]
    # Fresh LLM, agent and registry per run: nothing is shared across threads
    # except the pure tool functions themselves.
    call_log: list = []
    try:
        if hierarchical:
            registry = build_main_registry(
                llm_factory=lambda: _make_llm(provider),
                call_log=call_log,
                max_steps=sub_max_steps,
                subagent_system_prompt=subagent_prompt,
            )
        else:
            registry = ToolRegistry(ALL_TOOLS)
        agent = ReActAgent(
            llm=_make_llm(provider), tools=registry,
            system_prompt=system_prompt, max_steps=max_steps,
        )
        outcome = agent.run(task["prompt"])
    except Exception as exc:  # noqa: BLE001 - keep the sweep going
        with _PRINT_LOCK:
            print(f"  {task['id']:<16} t{trial} ERROR: {exc}", flush=True)
        return {
            "task_id": task["id"], "trial": trial, "error": str(exc),
            "exact_sequence_match": False, "answer_ok": False,
            "failure_kind": "error", "step_recall": 0.0, "extra_calls": 0,
            "delegate_calls": len(call_log), "triggers": task["_triggers"],
        }

    if hierarchical:
        # Flatten every specialist's own tool calls, in delegation order --
        # the same metric the flat conditions use, kept apples-to-apples.
        called = [n for entry in call_log for n in entry["called_tools"]]
    else:
        called = [
            (step.get("action") or {}).get("name")
            for step in outcome.trajectory if step.get("action")
        ]
    exact = called == expected
    answer_ok = all(
        s.lower() in outcome.answer.lower()
        for s in task.get("expect_substrings", [])
    )
    kind = classify_failure(called, expected)
    recall, extra = step_recall(called, expected)

    missing = list((Counter(expected) - Counter(called)).elements())
    with _PRINT_LOCK:
        print(f"  {task['id']:<16} t{trial} seq={'ok' if exact else 'X ':<2} "
              f"ans={'ok' if answer_ok else 'X ':<2} {kind:<6} "
              f"recall={recall:.2f} {len(called)}/{len(expected)}"
              f"{f'  deleg={len(call_log)}' if hierarchical else ''}"
              f"{'  missing=' + ','.join(missing) if missing else ''}", flush=True)

    if delay:
        time.sleep(delay)
    return {
        "task_id": task["id"], "trial": trial,
        "triggers": task["_triggers"],
        "delegate_calls": len(call_log),
        # Kept for hierarchical runs so a missing required call can be
        # localised: a tool whose group was never delegated to means the main
        # agent under-delegated; a tool whose group *was* delegated but that
        # never got called means the specialist dropped it.
        "call_log": [
            {"group": e["group"], "task": e["task"], "called_tools": e["called_tools"]}
            for e in call_log
        ] if hierarchical else None,
        "expect_tool_sequence": expected,
        "called_sequence": called,
        "exact_sequence_match": exact,
        "answer_ok": answer_ok,
        "failure_kind": kind,
        "step_recall": recall,
        "extra_calls": extra,
        "answer": outcome.answer,
        "steps": outcome.steps,
        "stop_reason": outcome.stop_reason,
    }


def run_condition(
    name: str, system_prompt: str, tasks: list[dict],
    provider: str, max_steps: int, delay: float, repeat: int, workers: int,
) -> dict:
    print("=" * 78)
    print(f"CONDITION: {name}   (provider={provider}, tools={len(ALL_TOOLS)}, "
          f"tasks={len(tasks)}, repeat={repeat}, workers={workers})")
    print("=" * 78)

    jobs = [(t, k) for t in tasks for k in range(repeat)]
    rows: list[dict] = []
    started = time.time()
    is_hier = name in HIERARCHICAL
    sub_prompt = SUBAGENT_PROMPTS.get(name)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_one_run, t, k, system_prompt, provider, max_steps,
                        delay, is_hier, 8, sub_prompt)
            for t, k in jobs
        ]
        for fut in as_completed(futures):
            rows.append(fut.result())
    rows.sort(key=lambda r: (r["task_id"], r["trial"]))
    print(f"  ({len(rows)} runs in {time.time() - started:.0f}s)")

    return summarize(name, system_prompt, rows)


def summarize(name: str, system_prompt: str, rows: list[dict]) -> dict:
    total = len(rows)
    exact = sum(r["exact_sequence_match"] for r in rows)
    answers = sum(r["answer_ok"] for r in rows)
    kinds = Counter(r["failure_kind"] for r in rows)
    mean_recall = sum(r["step_recall"] for r in rows) / total if total else 0.0
    extra_total = sum(r["extra_calls"] for r in rows)

    by_trigger: dict[str, dict] = defaultdict(lambda: {"n": 0, "exact": 0, "recall": 0.0})
    for row in rows:
        for tag in row["triggers"]:
            by_trigger[tag]["n"] += 1
            by_trigger[tag]["exact"] += int(row["exact_sequence_match"])
            by_trigger[tag]["recall"] += row["step_recall"]
    trigger_stats = {
        tag: {
            "n": v["n"], "exact": v["exact"],
            "accuracy": v["exact"] / v["n"] if v["n"] else 0.0,
            "mean_step_recall": v["recall"] / v["n"] if v["n"] else 0.0,
        }
        for tag, v in sorted(by_trigger.items())
    }

    # Per-task pass rate across trials -- the unit for paired tests when k>1.
    per_task: dict[str, list] = defaultdict(list)
    for row in rows:
        per_task[row["task_id"]].append(row["exact_sequence_match"])
    task_rates = {t: sum(v) / len(v) for t, v in sorted(per_task.items())}

    # Which required tool gets dropped, and how often.
    skipped = Counter()
    for row in rows:
        if row.get("called_sequence") is not None:
            for name_ in (Counter(row["expect_tool_sequence"])
                          - Counter(row["called_sequence"])).elements():
                skipped[name_] += 1

    print("-" * 78)
    print(f"  exact tool-sequence match : {exact}/{total} = {exact / total:.1%}")
    print(f"  mean step recall          : {mean_recall:.3f}")
    print(f"  final-answer accuracy     : {answers}/{total} = {answers / total:.1%}")
    print(f"  failure kinds             : {dict(kinds)}   extra calls: {extra_total}")
    delegs = [r.get("delegate_calls", 0) for r in rows]
    if any(delegs):
        print(f"  mean delegations per run  : {sum(delegs) / len(delegs):.2f}")
    print("  by trigger class:")
    for tag, v in trigger_stats.items():
        print(f"    {tag:<24} {v['exact']:>4}/{v['n']:<4} = {v['accuracy']:>6.1%}"
              f"   recall={v['mean_step_recall']:.3f}")
    if skipped:
        print("  most-skipped required tools:")
        for name_, c in skipped.most_common(6):
            print(f"    {name_:<32} {c}")
    print("=" * 78 + "\n")

    return {
        "condition": name,
        "system_prompt": system_prompt,
        "total": total,
        "exact_sequence_matches": exact,
        "sequence_accuracy": exact / total if total else 0.0,
        "mean_step_recall": mean_recall,
        "answer_accuracy": answers / total if total else 0.0,
        "failure_kinds": dict(kinds),
        "extra_calls_total": extra_total,
        "by_trigger_class": trigger_stats,
        "per_task_pass_rate": task_rates,
        "skipped_tools": dict(skipped.most_common()),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt-intervention ladder vs. silent step skipping.")
    parser.add_argument("--provider", default="deepseek", choices=["deepseek", "bailian", "openai"])
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        choices=list(CONDITIONS))
    parser.add_argument("--repeat", type=int, default=1, help="Runs per task per condition.")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_FILES),
                        help="Task JSON files (default: the 21-task Test-A set).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Concurrent agent runs. >1 is required for large sweeps.")
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--dump", metavar="PATH", default=None)
    args = parser.parse_args()

    tasks = _load_tasks(tuple(args.tasks))
    known = {t.name for t in ALL_TOOLS}
    for task in tasks:
        unknown = [t for t in task["expect_tool_sequence"] if t not in known]
        if unknown:
            raise RuntimeError(f"Task {task['id']} references unknown tools: {unknown}")

    report = {
        "provider": args.provider,
        "registry_size": len(ALL_TOOLS),
        "task_files": [os.path.basename(p) for p in args.tasks],
        "task_count": len(tasks),
        "chain_length": sorted({len(t["expect_tool_sequence"]) for t in tasks}),
        "repeat": args.repeat,
        "conditions": {},
    }

    for name in args.conditions:
        report["conditions"][name] = run_condition(
            name, CONDITIONS[name], tasks,
            args.provider, args.max_steps, args.delay, args.repeat, args.workers,
        )
        if args.dump:  # write after every condition so a crash keeps progress
            with open(args.dump, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=2, ensure_ascii=False)

    print("=" * 78)
    print("LADDER SUMMARY")
    print("=" * 78)
    print(f"  {'condition':<14} {'seq match':>10} {'recall':>8} {'answer':>8} "
          f"{'extra':>6}   failure kinds")
    for name, res in report["conditions"].items():
        print(f"  {name:<14} {res['sequence_accuracy']:>9.1%} "
              f"{res['mean_step_recall']:>8.3f} {res['answer_accuracy']:>8.1%} "
              f"{res['extra_calls_total']:>6}   {res['failure_kinds']}")
    print("=" * 78)

    if args.dump:
        print(f"\nWrote full results to {args.dump}")


if __name__ == "__main__":
    main()
