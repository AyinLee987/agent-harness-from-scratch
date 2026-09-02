# Intervention ladder: a one-line prompt change beats the architectural fix

**Date:** 2026-09-02 · **Model:** `deepseek-chat` · **Tools:** 100 · **Tasks:** 21 five-step (Test-A)
**Script:** `examples/intervention_ladder_test.py` · **Raw:** `examples/intervention_ladder_results.json`

## Why this ran

The tool-scaling experiments found that at 5+ step chains the exact tool-sequence
match rate falls to 57% while final-answer accuracy stays at 100% — the model
silently skips steps it judges redundant. Hierarchical routing then recovered
57% → 67% at ~2x the LLM round trips.

Nobody had checked the cheapest possible intervention first. That mattered,
because `DEFAULT_SYSTEM_PROMPT` ends with:

> "When you have enough information, respond with a final answer and do not call any more tools."

The shipped prompt was actively encouraging the exact behaviour being measured
as a failure.

## Results

| Condition | Exact sequence match | Wilson 95% CI | Answer accuracy | Failure kinds |
|---|---|---|---|---|
| historical (2026-08) | 12/21 = 57.1% | [36.5%, 75.5%] | 100% | 9 skip |
| **baseline** (re-run, same prompt) | 15/21 = **71.4%** | [50.0%, 86.2%] | 100% | 6 skip |
| **instruction** | 19/21 = **90.5%** | [71.1%, 97.3%] | 100% | 2 skip |
| **fewshot** | 19/21 = **90.5%** | [71.1%, 97.3%] | 100% | 2 skip |
| *hierarchical routing (prior work)* | *14/21 = 67%* | — | *100%* | — |

Paired McNemar (exact, two-sided), all discordant pairs in one direction:

| Comparison | Improved / regressed | p |
|---|---|---|
| historical → instruction | +7 / −0 | **0.016** |
| historical → baseline | +3 / −0 | 0.250 |
| baseline → instruction | +4 / −0 | 0.125 |
| instruction → fewshot | +0 / −0 | 1.000 |

## Four findings

### 1. The baseline moved 14 points with nothing changed

Same prompt, same model, same 21 tasks: 57.1% in August, **71.4%** today. Three
tasks flipped to passing, none regressed. Nothing in the configuration changed.

**This is the most consequential result here**, because it means the headline
57% → 67% hierarchical-routing improvement is not established: 67% sits inside
the baseline's own run-to-run band, and today's untouched baseline (71.4%) is
*above* it. The 21-task set cannot resolve a 10-point effect. `PLAN.md` §3
constraint B predicted exactly this ("每个任务值 4.76 个百分点，57%→67% 实际只
相差 2 个任务"); this is the empirical demonstration.

### 2. One instruction recovers more than the architecture did, at zero cost

Replacing the early-stop clause with an explicit completeness rule took the same
day's baseline from 71.4% to 90.5% — 4 tasks fixed, 0 broken. Against the
historical baseline the effect clears significance (p=0.016); against the
same-day baseline it does not (p=0.125), because there were only 4 discordant
pairs to test with.

Either way the cost comparison is stark: hierarchical routing bought its points
with ~2x LLM round trips and a new failure surface (cross-specialist duplicate
delegation). This bought more of them with one paragraph and no runtime cost.

### 3. Few-shot examples added exactly nothing

`instruction` and `fewshot` produced **identical per-task results** — 0
discordant pairs. Two worked examples demonstrating a no-op round and an
identity round trip changed no outcome the instruction had not already changed.
The rule was sufficient; the demonstrations were redundant.

### 4. No over-correction at all

`extra` (every required call made, plus unrequested ones) was **0 in all three
conditions**. The feared failure mode of "train/prompt the model to never skip
and it starts calling tools it shouldn't" did not appear. Final-answer accuracy
stayed 100% throughout. This is a Test-C-style signal for free, though on tasks
whose expected sequence is exact — a proper over-correction test still needs
tasks with genuinely optional steps.

## What survives every intervention

Two tasks fail under all three conditions, both at maximum redundancy pressure:

| Task | Missing | Why it is hard |
|---|---|---|
| `celsius_roundtrip` | `math_round` | 100°C → F → C → F → C returns exactly 100. Rounding an integer to the nearest integer is a *pure* no-op, sitting at the end of an identity round trip (T1 + T4 + T5 stacked). |
| `add_days_twice` | `calendar_is_weekend`, `calendar_week_number` | Once the model has the weekday name, both remaining facts are inferable without a tool (T2 + T4). |

Every single failure across every condition was `skip` — never a wrong tool,
never a wrong order. The diagnosis from the original experiment holds.

Skipped-step distribution: `math_round` in 4 of 6 baseline failures and 1 of 2
residual failures; `calendar_is_weekend` in the rest. T1 (no-op round) and T2
(inferable boolean) are the entire failure surface at this chain length.

## Consequences

1. **The resume line "精确匹配率 57% → 67%" should not stand as written.** The
   effect is inside the noise of a 21-task set. Either re-measure on ~200 tasks
   with CIs, or restate as a cost comparison rather than an accuracy gain.
2. **Any weight-level work must clear 90.5%, not 57%.** The realistic target for
   a trained guard or a fine-tune is now the 2 residual tasks, not the 9 the
   project started from.
3. **Test-B is now the blocking prerequisite**, not an improvement. At n=21 the
   95% CIs of every condition overlap; nothing here can be called significant
   with confidence. `PLAN.md` §3 constraint B already required ~200 held-out
   tasks — that work has to happen before any further intervention is evaluated.

## Reproduce

```bash
python examples/intervention_ladder_test.py --dump examples/intervention_ladder_results.json
# ~63 agent runs, ~380 API calls, ~$0.2 on deepseek-chat
```

## Threats to validity

- **k=1.** One run per task per condition. Given finding 1, single-sample
  results on this task set are demonstrably unstable; k≥5 is needed before any
  of these numbers should be quoted.
- **n=21.** Below the resolution needed for the effects being measured.
- **Few-shot demonstrations are pattern-adjacent.** They use tools absent from
  Test-A, but they do demonstrate the T1 and T5 patterns. Since few-shot changed
  nothing, this caveat is moot here — it would matter if it had helped.
- **`instruction` changes two things at once**: it removes the early-stop clause
  *and* adds the completeness rule. Which half does the work is untested; an
  ablation removing only the early-stop clause is the obvious next run and is
  nearly free.
