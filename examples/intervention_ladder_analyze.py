"""Paired statistics over the intervention-ladder result dumps.

Deliberately reuses the companion evaluation project's statistics engine
(``agent_eval.stats``) rather than reimplementing bootstrap here -- that
package exists precisely so results in this repo come with intervals instead
of bare point estimates.

The unit of analysis is a *task*, not a run: with k trials per task the
per-task pass rate (0, 1/3, 2/3, 1) is a continuous score, and conditions are
compared as matched pairs over the same 60 tasks. Comparing runs instead
would treat three trials of the same task as independent, which they are not.

Usage
-----
    python examples/intervention_ladder_analyze.py \
        examples/intervention_ladder_8step_results.json \
        examples/intervention_ladder_8step_results_part2.json \
        examples/intervention_ladder_8step_hier.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

EVAL_KIT = r"C:\Users\Li Zhuoyang\Desktop\evaluation"
if os.path.isdir(EVAL_KIT):
    sys.path.insert(0, EVAL_KIT)

from agent_eval.stats import bootstrap_ci, paired_bootstrap_test  # noqa: E402

BASELINE = "baseline"


def load(paths: list[str]) -> dict[str, dict]:
    conditions: dict[str, dict] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        for name, res in blob["conditions"].items():
            errors = sum(1 for r in res["rows"] if r.get("error"))
            if errors:
                print(f"  ! skipping {name} from {os.path.basename(path)}: "
                      f"{errors}/{res['total']} runs are API errors, not measurements")
                continue
            conditions[name] = res
    return conditions


def per_task(res: dict, field: str) -> dict[str, float]:
    acc: dict[str, list] = defaultdict(list)
    for row in res["rows"]:
        acc[row["task_id"]].append(float(row[field]))
    return {t: sum(v) / len(v) for t, v in acc.items()}


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        print(__doc__)
        raise SystemExit(2)

    print("Loading result dumps")
    conditions = load(paths)
    order = [c for c in ["baseline", "instruction", "fewshot", "hierarchical",
                         "hierarchical_instruction"] if c in conditions]
    print(f"  conditions: {order}\n")

    tasks = sorted(per_task(conditions[BASELINE], "exact_sequence_match"))
    print(f"Unit of analysis: {len(tasks)} tasks, "
          f"k={conditions[BASELINE]['total'] // len(tasks)} trials each\n")

    for field, label in [("exact_sequence_match", "EXACT TOOL-SEQUENCE MATCH"),
                         ("step_recall", "STEP RECALL"),
                         ("answer_ok", "FINAL-ANSWER ACCURACY")]:
        print("=" * 76)
        print(label)
        print("=" * 76)
        base_scores = per_task(conditions[BASELINE], field)
        base_vec = [base_scores[t] for t in tasks]
        print(f"  {'condition':<26} {'mean':>7}  {'95% CI':>16}  "
              f"{'vs baseline':>12}  {'p':>8}")
        for name in order:
            scores = per_task(conditions[name], field)
            vec = [scores[t] for t in tasks]
            ci = bootstrap_ci(vec, seed=0)
            if name == BASELINE:
                print(f"  {name:<26} {ci.mean:>7.3f}  "
                      f"[{ci.low:.3f}, {ci.high:.3f}]{'':>3}  {'--':>12}  {'--':>8}")
                continue
            test = paired_bootstrap_test(base_vec, vec, seed=0)
            delta = test.mean_diff
            star = " *" if test.p_value < 0.05 else ""
            print(f"  {name:<26} {ci.mean:>7.3f}  "
                  f"[{ci.low:.3f}, {ci.high:.3f}]{'':>3}  "
                  f"{delta:>+12.3f}  {test.p_value:>8.4f}{star}")
        print()

    print("=" * 76)
    print("FAILURE COMPOSITION  (runs, not tasks)")
    print("=" * 76)
    print(f"  {'condition':<26} {'match':>6} {'skip':>6} {'extra':>6} {'other':>6} "
          f"{'extra calls':>12}")
    for name in order:
        k = conditions[name]["failure_kinds"]
        print(f"  {name:<26} {k.get('match', 0):>6} {k.get('skip', 0):>6} "
              f"{k.get('extra', 0):>6} {k.get('other', 0):>6} "
              f"{conditions[name]['extra_calls_total']:>12}")
    print()

    print("=" * 76)
    print("EXACT MATCH BY TRIGGER CLASS")
    print("=" * 76)
    tags = sorted(conditions[BASELINE]["by_trigger_class"])
    print(f"  {'trigger':<24}" + "".join(f"{n[:12]:>14}" for n in order))
    for tag in tags:
        cells = ""
        for name in order:
            s = conditions[name]["by_trigger_class"].get(tag)
            cells += f"{s['accuracy']:>13.1%} " if s else f"{'--':>14}"
        n = conditions[BASELINE]["by_trigger_class"][tag]["n"]
        print(f"  {tag:<20}n={n:<3}" + cells)
    print()

    print("=" * 76)
    print("COST")
    print("=" * 76)
    for name in order:
        rows = conditions[name]["rows"]
        delegs = [r.get("delegate_calls", 0) for r in rows]
        steps = [r.get("steps", 0) for r in rows]
        print(f"  {name:<26} mean agent steps={sum(steps) / len(steps):>5.1f}   "
              f"mean delegations={sum(delegs) / len(delegs):>5.2f}")


if __name__ == "__main__":
    main()
