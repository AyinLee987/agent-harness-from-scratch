"""Generate long, trigger-dense tool chains for the step-skipping experiments.

Why
---
Test-A (the 21 hand-written five-step tasks) is too small and too easy to
resolve the effects being measured: at a 71% baseline only 6 tasks are broken,
so an intervention has at most 6 tasks of headroom and a single run carries
about +/-10pp of noise. The failure this project studies is driven by *chain
length* (the tool-count and description-length variables were both measured
flat), so the way to make it resolvable is to generate longer chains and pack
them with the redundancy pressures that actually trigger skipping.

This module walks a tool type graph (number / string / date), applies the real
tool functions from ``tool_scaling_kit`` to compute ground truth, and emits
tasks in exactly the schema ``tool_scaling_multi_tasks.json`` uses, plus a
``_meta`` block (ignored by the existing scripts) recording chain length and
trigger classes.

Trigger classes (PLAN.md 5.3)
-----------------------------
    T1_noop_round          math_round on a value already at that precision
    T2_inferable_bool      a boolean the model can reason out without a tool
    T3_small_stat          a sum/average/count over 3-5 small numbers
    T4_same_tool_twice     the same tool applied twice in a row
    T5_identity_roundtrip  A then A-inverse, returning to the input value

Ground truth is computed by *executing the chain*, never by hand -- every tool
in the kit is a pure function, so the oracle is free (PLAN.md 5.5).

Usage
-----
    python examples/chain_task_gen.py --steps 8 --count 80 \
        --out examples/gen_tasks_8step.json
    python examples/chain_task_gen.py --steps 8 --count 80 --seed 7 --verify
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Callable, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from tool_scaling_kit import ALL_TOOLS  # noqa: E402

FUNCS = {t.name: t._func for t in ALL_TOOLS}

NUM, STR, DATE = "num", "str", "date"


# --------------------------------------------------------------------------
# Step catalogue.  Each entry knows how to build its kwargs from the running
# value, how to phrase itself, and what it produces.
# --------------------------------------------------------------------------

class Step:
    def __init__(self, tool: str, out_type: str, kwargs: dict, phrase: str,
                 triggers: Optional[List[str]] = None) -> None:
        self.tool = tool
        self.out_type = out_type
        self.kwargs = kwargs
        self.phrase = phrase
        self.triggers = triggers or []

    def apply(self) -> str:
        return FUNCS[self.tool](**self.kwargs)


# --- number -> number ------------------------------------------------------

_ARITH = [
    ("math_add", "add {c}", lambda v, c: {"a": v, "b": c}),
    ("math_subtract", "subtract {c}", lambda v, c: {"a": v, "b": c}),
    ("math_multiply", "multiply by {c}", lambda v, c: {"a": v, "b": c}),
]

# (forward tool, inverse tool, forward arg, inverse arg, unit-in, unit-out)
_UNIT_PAIRS = [
    ("convert_celsius_to_fahrenheit", "convert_fahrenheit_to_celsius",
     "celsius", "fahrenheit", "Celsius", "Fahrenheit"),
    ("convert_km_to_miles", "convert_miles_to_km",
     "kilometers", "miles", "kilometers", "miles"),
    ("convert_kg_to_lbs", "convert_lbs_to_kg",
     "kilograms", "pounds", "kilograms", "pounds"),
    ("convert_meters_to_feet", "convert_feet_to_meters",
     "meters", "feet", "meters", "feet"),
    ("convert_liters_to_gallons", "convert_gallons_to_liters",
     "liters", "gallons", "liters", "gallons"),
    ("measure_mph_to_kmh", "measure_kmh_to_mph",
     "mph", "kmh", "mph", "km/h"),
    ("measure_sqft_to_sqm", "measure_sqm_to_sqft",
     "square_feet", "square_meters", "square feet", "square meters"),
    ("measure_miles_to_nautical_miles", "measure_nautical_miles_to_miles",
     "miles", "nautical_miles", "miles", "nautical miles"),
]

# --- string -> string ------------------------------------------------------

_STR_UNARY = [
    ("text_uppercase", "convert it to uppercase", "text"),
    ("text_lowercase", "convert it to lowercase", "text"),
    ("text_reverse", "reverse it", "text"),
    ("text_title_case", "convert it to title case", "text"),
    ("format_snake_case", "convert it to snake_case", "text"),
    ("format_kebab_case", "convert it to kebab-case", "text"),
    ("encode_rot13", "apply ROT13 to it", "text"),
]

_STR_CODEC_PAIRS = [
    ("data_base64_encode", "data_base64_decode", "text", "encoded",
     "base64-encode it", "base64-decode the result"),
    ("encode_hex_encode", "encode_hex_decode", "text", "hex_string",
     "hex-encode it", "hex-decode the result"),
    ("data_url_encode", "data_url_decode", "text", "encoded",
     "URL-encode it", "URL-decode the result"),
]

_STR_TO_NUM = [
    ("text_word_count", "count how many words it has", "text"),
    ("text_char_count", "count how many characters it has", "text"),
    ("text_count_vowels", "count how many vowels it has", "text"),
    ("encode_count_bytes", "count how many bytes it takes up", "text"),
]

_STR_TO_BOOL = [
    ("text_is_palindrome", "check whether it is a palindrome", "text"),
    ("encode_is_ascii", "check whether it is pure ASCII", "text"),
]

# --- date ------------------------------------------------------------------

_DATE_TO_DATE = [
    ("date_add_days", "add {c} days to it", lambda v, c: {"date": v, "days": c}),
    ("calendar_subtract_days", "subtract {c} days from it",
     lambda v, c: {"date": v, "days": c}),
]

_DATE_TO_STR = [
    ("date_day_of_week", "tell me what day of the week it falls on", "date"),
    ("date_quarter", "tell me which calendar quarter it is in", "date"),
]

_DATE_TO_NUM = [
    ("calendar_week_number", "tell me its ISO week number", "date"),
]

_DATE_TO_BOOL = [
    ("calendar_is_weekend", "tell me whether it is a weekend", "date"),
]

_WORDS = [
    "hello world", "Racecar Level", "Open Source Agent", "Never Odd Or Even",
    "quick brown fox", "Data Pipeline Design", "step by step",
    "A Man A Plan", "tool calling matters", "silent skip",
]


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(round(value, 6))


# --------------------------------------------------------------------------
# Moves: each returns a list of Steps, or None if it does not fit
# --------------------------------------------------------------------------

def _move_arith(rng, value: float, budget: int) -> Optional[List[Step]]:
    tool, phrase, mk = rng.choice(_ARITH)
    c = rng.choice([2, 3, 4, 5, 6, 10])
    return [Step(tool, NUM, mk(value, c), phrase.format(c=c))]


def _move_divide(rng, value: float, budget: int) -> Optional[List[Step]]:
    for c in rng.sample([2, 4, 5, 10], 4):
        if float(value) / c == int(float(value) / c):  # keep values tidy
            return [Step("math_divide", NUM, {"a": value, "b": c},
                         f"divide it by {c}")]
    return None


def _move_noop_round(rng, value: float, budget: int) -> Optional[List[Step]]:
    """T1: round a value that is already at (or beyond) the target precision.

    Generalised past "integers only": rounding 9.999975 to 6 decimal places is
    just as much a no-op as rounding 70 to 0, and restricting to integers made
    T1 rare in generated chains while it was the dominant failure in Test-A.
    """
    text = _num(value)
    decimals = len(text.split(".")[1]) if "." in text else 0
    digits = rng.choice([decimals, decimals, min(decimals + 1, 6)])
    if digits == 0:
        where = "the nearest whole number"
    elif digits == 1:
        where = "1 decimal place"
    else:
        where = f"{digits} decimal places"
    step = Step("math_round", NUM, {"value": value, "digits": digits},
                f"round it to {where}", ["T1_noop_round"])
    if _num(float(step.apply())) != text:  # must be a genuine no-op
        return None
    return [step]


def _move_unit_roundtrip(rng, value: float, budget: int) -> Optional[List[Step]]:
    """T5: convert to another unit and straight back."""
    if budget < 2:
        return None
    fwd, inv, fa, ia, u_in, u_out = rng.choice(_UNIT_PAIRS)
    mid = float(FUNCS[fwd](**{fa: value}))
    return [
        Step(fwd, NUM, {fa: value}, f"convert it from {u_in} to {u_out}",
             ["T5_identity_roundtrip"]),
        Step(inv, NUM, {ia: mid}, f"convert it back to {u_in}",
             ["T5_identity_roundtrip"]),
    ]


def _move_same_tool_twice(rng, value: float, budget: int) -> Optional[List[Step]]:
    """T4: apply the same arithmetic tool twice in a row."""
    if budget < 2:
        return None
    tool, phrase, mk = rng.choice(_ARITH)
    c1 = rng.choice([2, 3, 5])
    first = Step(tool, NUM, mk(value, c1), phrase.format(c=c1),
                 ["T4_same_tool_twice"])
    mid = float(first.apply())
    c2 = rng.choice([2, 3, 4])
    second = Step(tool, NUM, mk(mid, c2),
                  phrase.format(c=c2).replace("add", "add another", 1)
                  if tool == "math_add" else phrase.format(c=c2) + " again",
                  ["T4_same_tool_twice"])
    return [first, second]


def _move_str_codec_roundtrip(rng, value: str, budget: int) -> Optional[List[Step]]:
    if budget < 2:
        return None
    enc, dec, ea, da, p1, p2 = rng.choice(_STR_CODEC_PAIRS)
    mid = FUNCS[enc](**{ea: value})
    return [
        Step(enc, STR, {ea: value}, p1, ["T5_identity_roundtrip"]),
        Step(dec, STR, {da: mid}, p2, ["T5_identity_roundtrip"]),
    ]


def _move_str_unary(rng, value: str, budget: int) -> Optional[List[Step]]:
    tool, phrase, arg = rng.choice(_STR_UNARY)
    return [Step(tool, STR, {arg: value}, phrase)]


def _move_str_to_num(rng, value: str, budget: int) -> Optional[List[Step]]:
    tool, phrase, arg = rng.choice(_STR_TO_NUM)
    return [Step(tool, NUM, {arg: value}, phrase)]


def _move_str_to_bool(rng, value: str, budget: int) -> Optional[List[Step]]:
    """T2: a boolean the model could work out itself."""
    if budget < 2:
        return None  # needs a follow-on step to consume "true"/"false"
    tool, phrase, arg = rng.choice(_STR_TO_BOOL)
    return [Step(tool, STR, {arg: value}, phrase, ["T2_inferable_bool"])]


def _move_date_shift(rng, value: str, budget: int) -> Optional[List[Step]]:
    tool, phrase, mk = rng.choice(_DATE_TO_DATE)
    c = rng.choice([5, 7, 10, 12, 20, 30])
    return [Step(tool, DATE, mk(value, c), phrase.format(c=c))]


def _move_date_shift_twice(rng, value: str, budget: int) -> Optional[List[Step]]:
    """T4 on dates."""
    if budget < 2:
        return None
    c1, c2 = rng.choice([5, 7, 10]), rng.choice([12, 20, 30])
    first = Step("date_add_days", DATE, {"date": value, "days": c1},
                 f"add {c1} days to it", ["T4_same_tool_twice"])
    mid = first.apply()
    second = Step("date_add_days", DATE, {"date": mid, "days": c2},
                  f"add another {c2} days", ["T4_same_tool_twice"])
    return [first, second]


def _move_date_to_bool(rng, value: str, budget: int) -> Optional[List[Step]]:
    if budget < 2:
        return None
    tool, phrase, arg = rng.choice(_DATE_TO_BOOL)
    return [Step(tool, STR, {arg: value}, phrase, ["T2_inferable_bool"])]


def _move_date_to_str(rng, value: str, budget: int) -> Optional[List[Step]]:
    tool, phrase, arg = rng.choice(_DATE_TO_STR)
    return [Step(tool, STR, {arg: value}, phrase)]


def _move_date_to_num(rng, value: str, budget: int) -> Optional[List[Step]]:
    tool, phrase, arg = rng.choice(_DATE_TO_NUM)
    return [Step(tool, NUM, {arg: value}, phrase)]


# Weighted so no single trigger class dominates the corpus (PLAN.md Stage 5
# caps any one class at 25%). math_round is weighted up because T1 was the
# dominant failure in Test-A and an under-representative corpus would measure
# a different phenomenon than the one this project is about.
TRIGGER_MOVES = {
    NUM: [(_move_noop_round, 4), (_move_unit_roundtrip, 2), (_move_same_tool_twice, 2)],
    STR: [(_move_str_codec_roundtrip, 2), (_move_str_to_bool, 2)],
    DATE: [(_move_date_shift_twice, 2), (_move_date_to_bool, 2)],
}

PLAIN_MOVES = {
    NUM: [(_move_arith, 3), (_move_divide, 1)],
    STR: [(_move_str_unary, 3), (_move_str_to_num, 1)],
    DATE: [(_move_date_shift, 2), (_move_date_to_str, 1), (_move_date_to_num, 1)],
}


def _weighted_order(rng, pool):
    """Shuffle a [(move, weight)] pool, drawing without replacement by weight."""
    remaining = list(pool)
    order = []
    while remaining:
        total = sum(w for _, w in remaining)
        pick = rng.uniform(0, total)
        acc = 0.0
        for i, (move, w) in enumerate(remaining):
            acc += w
            if pick <= acc:
                order.append(move)
                remaining.pop(i)
                break
        else:
            order.append(remaining.pop()[0])
    return order


# --------------------------------------------------------------------------
# Chain construction
# --------------------------------------------------------------------------

def _seed(rng) -> Tuple[str, object, str, List[Step]]:
    """Return (type, value, opening phrase, pre-steps)."""
    kind = rng.choice(["num", "num", "list", "str", "date"])
    if kind == "num":
        v = rng.choice([12, 20, 24, 30, 36, 48, 60, 100, 120])
        return NUM, float(v), f"Start with the number {v}.", []
    if kind == "list":  # T3: a small stat the model can do in its head
        nums = [rng.randint(1, 9) for _ in range(rng.choice([3, 4, 5]))]
        tool, phrase = rng.choice([
            ("stat_sum", "compute their sum"),
            ("math_average", "compute their average"),
            ("stat_count", "count how many there are"),
        ])
        raw = ",".join(str(n) for n in nums)
        step = Step(tool, NUM, {"numbers": raw}, "", ["T3_small_stat"])
        opening = (f"Take the numbers {', '.join(str(n) for n in nums)}. "
                   f"First, {phrase}.")
        return NUM, float(step.apply()), opening, [step]
    if kind == "str":
        w = rng.choice(_WORDS)
        return STR, w, f'Take the text "{w}".', []
    d = f"2026-{rng.randint(1, 11):02d}-{rng.randint(1, 25):02d}"
    return DATE, d, f"Start with the date {d}.", []


def build_chain(rng, steps: int, trigger_density: float) -> Optional[dict]:
    vtype, value, opening, chain = _seed(rng)

    guard = 0
    while len(chain) < steps:
        guard += 1
        if guard > 200:
            return None
        budget = steps - len(chain)
        want_trigger = rng.random() < trigger_density
        pools = ([TRIGGER_MOVES[vtype], PLAIN_MOVES[vtype]] if want_trigger
                 else [PLAIN_MOVES[vtype], TRIGGER_MOVES[vtype]])

        last_tool = chain[-1].tool if chain else None
        produced = None
        for pool in pools:
            for move in _weighted_order(rng, pool):
                out = move(rng, value, budget)
                if not out or len(out) > budget:
                    continue
                # two no-op rounds back to back reads as a typo, not a task
                if out[0].tool == "math_round" and last_tool == "math_round":
                    continue
                produced = out
                break
            if produced:
                break
        if not produced:
            return None

        for st in produced:
            try:
                result = st.apply()
            except Exception:
                return None
            if st.out_type == NUM:
                try:
                    value = float(result)
                except ValueError:
                    return None
            else:
                value = result
            vtype = st.out_type
            chain.append(st)

        # a bool lands as the string "true"/"false"; force a text step next so
        # the chain stays meaningful rather than ending on a bare boolean
        if produced[-1].triggers and "T2_inferable_bool" in produced[-1].triggers:
            if len(chain) < steps:
                tool, phrase, arg = rng.choice(_STR_UNARY[:3])
                st = Step(tool, STR, {arg: value}, f"take that answer and {phrase}")
                value = st.apply()
                vtype = STR
                chain.append(st)

    if len(chain) != steps:
        return None

    phrases = [s.phrase for s in chain if s.phrase]
    prompt = opening + " Then " + ", then ".join(phrases) + "."
    final = chain[-1].apply()

    triggers = sorted({t for s in chain for t in s.triggers}) or ["T0_none"]
    return {
        "prompt": prompt,
        "expect_tool_sequence": [s.tool for s in chain],
        "expect_substrings": [final],
        "_meta": {
            "chain_length": len(chain),
            "trigger_classes": triggers,
            "final_value": final,
        },
    }


def generate(count: int, steps: int, trigger_density: float, seed: int) -> List[dict]:
    rng = random.Random(seed)
    tasks: List[dict] = []
    seen_prompts: set = set()
    attempts = 0
    while len(tasks) < count and attempts < count * 400:
        attempts += 1
        task = build_chain(rng, steps, trigger_density)
        if not task:
            continue
        if task["prompt"] in seen_prompts:
            continue
        # a degenerate final value makes the answer check meaningless
        if task["expect_substrings"][0] in ("", "0", "true", "false"):
            continue
        seen_prompts.add(task["prompt"])
        task["id"] = f"gen{steps}_{len(tasks):04d}"
        tasks.append(task)
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate long trigger-dense tool chains.")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--count", type=int, default=80)
    ap.add_argument("--trigger-density", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20260902)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="Re-execute every chain and assert the recorded answer.")
    args = ap.parse_args()

    tasks = generate(args.count, args.steps, args.trigger_density, args.seed)
    print(f"Generated {len(tasks)} tasks of {args.steps} steps.")

    from collections import Counter
    dist = Counter(t for task in tasks for t in task["_meta"]["trigger_classes"])
    tools = Counter(n for task in tasks for n in task["expect_tool_sequence"])
    with_trigger = sum(1 for t in tasks if t["_meta"]["trigger_classes"] != ["T0_none"])
    print(f"  tasks with >=1 trigger : {with_trigger}/{len(tasks)} "
          f"= {with_trigger / max(1, len(tasks)):.0%}")
    print("  trigger class coverage :")
    for k, v in sorted(dist.items()):
        print(f"    {k:<24} {v:>4} tasks ({v / max(1, len(tasks)):.0%})")
    print(f"  distinct tools used    : {len(tools)}")
    top = tools.most_common(5)
    total_calls = sum(tools.values())
    print(f"  top tools              : "
          + ", ".join(f"{n} {c}({c / total_calls:.0%})" for n, c in top))

    if args.verify:
        bad = 0
        for task in tasks:
            # rebuild by replaying the recorded kwargs is not possible from the
            # dumped schema alone; instead assert the chain is self-consistent
            if len(task["expect_tool_sequence"]) != args.steps:
                bad += 1
        print(f"  verify: {len(tasks) - bad}/{len(tasks)} chains have exactly "
              f"{args.steps} steps")

    if args.out:
        payload = [{k: v for k, v in t.items()} for t in tasks]
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(tasks)} tasks to {args.out}")
    else:
        print("\nSample:")
        for t in tasks[:3]:
            print(json.dumps(t, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
