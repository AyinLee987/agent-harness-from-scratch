# Hierarchical routing eval: the one final-answer miss

`examples/hierarchical_agent_test.py`'s 21-task run (see README, "Hierarchical
routing: main agent + specialist subagents") scored final-answer accuracy at
95% (20/21) instead of the flat baseline's 100%. This note records which task
missed and why, so the dip isn't mistaken for a real regression.

## The task

**`lower_snake_reverse_count_leap`** (from `tool_scaling_long_chain_tasks.json`):

> Take the text 'Data Science Rocks'. Convert it to lowercase, then convert
> that to snake_case, then reverse the result, then count how many
> characters are in the reversed text, then check whether that character
> count is a leap year.

`expect_tool_sequence`: `text_lowercase → format_snake_case → text_reverse →
text_char_count → date_is_leap_year`
`expect_substrings`: `["false"]`

## What actually happened

Tool-sequence match was **exact** (`seq=ok`) — every specialist call landed
in the right group, in the right order:

```
text_agent     -> text_lowercase        -> 'data science rocks'
text_agent     -> format_snake_case     -> 'data_science_rocks'
text_agent     -> text_reverse          -> 'skcor_ecneics_atad'
text_agent     -> text_char_count       -> 18
datetime_agent -> date_is_leap_year     -> "The tool returned `false`,
                                            indicating that year 18 is
                                            not a leap year."
```

The main agent's final summary was:

> Let me compile the final result of the entire chain: ... **Leap year
> check:** Year 18 is **not** a leap year. **Final Answer:** The reversed
> text 'skcor_ecneics_atad' has **18 characters**, and the year **18 is not
> a leap year**.

Semantically correct — the model reasoned about the fact accurately and
threaded it through the whole chain. It just never emitted the literal
token `false`; it said "is not a leap year" instead. The substring grader
(`expect_substrings: ["false"]`) doesn't accept that paraphrase, so the row
scored `answer_ok=False`.

## Why this isn't a new defect

This is the exact same grading-artifact class already documented for the
flat-registry run of this same task (README, "All chains extended to 5+
steps" section) — a known limitation of substring-based grading used
throughout this repo's eval harness, not a model or tool-selection failure.
It reappeared here by coincidence (same task, same phrasing tendency), not
because hierarchical routing introduced anything new. The hierarchy's own
genuinely new failure mode in this run was a *different* task
(`add_days_then_weekday`'s redundant duplicate delegation — see README).

## Takeaway

Of the 21 tasks, this is the only place final-answer accuracy and
tool-sequence accuracy disagree in the hierarchical run. Net: 20/21 correct
answers, 21/21 semantically-correct answers if graded by a human instead of
a substring check.
