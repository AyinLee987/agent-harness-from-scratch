# Unifying the execution core (BUGS.md #22)

Plan for the one item from the 2026-09-05 external review that was **not**
patched, because patching it is the wrong move. Everything else from that
review is fixed and covered by `tests/test_review_fixes.py`.

Status: **not started.** This document is the design, not a record.

---

## 1. What is actually wrong

`/api/run` executes through `ReActLoop`. `/api/stream` re-implements the same
state machine inline, in `app/server.py::_stream_leader_steps`, so that it can
yield SSE events between phases.

That is one root cause with three measured symptoms — the review filed them
as three separate P1s, which overstates the count and understates the problem.

| # | Symptom | Measured |
|---|---------|----------|
| 1 | Sync `dispatcher.dispatch()` inside an `async` generator blocks the event loop | tool sleeps 300ms → a coroutine due at 30ms woke at **342ms** |
| 2 | Only `FatalToolError` is caught, so `SuspendRun` escapes | `uncaught_error=SuspendRun`, no `suspended` event, **0** checkpoints saved |
| 3 | Local `step_idx`, never `ctx.new_step()`; tokens count output text only | 2 model calls → `done.steps = 0`; ~1000-token prompt → `tokens = 5` |

The drift is not random. Forced reflection *was* ported to the streaming path,
with cross-referencing comments in both directions — and it is still correct
in both. Everything added without that discipline (step recording, suspension,
usage accounting, and now #16's per-call trajectory records) exists in one
copy only.

**The structural error is that streaming is an output concern being solved by
duplicating control flow.** Any fix that leaves two state machines in place
just resets the drift clock.

---

## 2. Target shape

`ReActLoop` becomes a generator of run events. Both endpoints consume the same
generator and differ only in transport.

```
                    ┌──────────────────────────────┐
                    │  ReActLoop.iter_run(task)    │
                    │  yields RunEvent, owns the   │
                    │  single ExecutionContext     │
                    └──────────────┬───────────────┘
                                   │
                ┌──────────────────┴──────────────────┐
                │                                     │
      /api/run: drain to the end             /api/stream: forward
      return the final AgentResult           each event as SSE
```

`run()` keeps its exact current signature and becomes a thin drain over
`iter_run()`, so nothing outside this module changes and every existing test
keeps passing unmodified. That is the property that makes the refactor
reviewable: if `run()`'s behaviour changes at all, something is wrong.

### The event set

Deliberately small, and mirroring what the SSE endpoint already emits so the
frontend does not have to change in the same commit:

| Event | Payload | Emitted by |
|-------|---------|-----------|
| `run_started` | `run_id`, `task` | `iter_run` |
| `think_started` | `step` | `_think_node` |
| `text` | `step`, `token` | streaming provider only |
| `think_completed` | `step`, `thought`, `usage` | `_think_node` |
| `tool_started` | `step`, `id`, `name`, `arguments` | `_act_node` |
| `tool_completed` | `step`, `id`, `name`, `observation`, `ok` | `_act_node` |
| `reflection` | `step`, `text` | `_think_node` |
| `suspended` | `run_id`, `pending_job_ids` | `_act_node` |
| `error` | `step`, `type`, `message` | any node |
| `run_completed` | `answer`, `stop_reason`, `steps`, `tokens` | `iter_run` |

`tool_started`/`tool_completed` map one-to-one onto #16's per-call trajectory
records — the same pairing, once, instead of once per transport.

---

## 3. Ordered steps

Each step is independently shippable and independently revertable. Steps 1-2
change no behaviour at all.

**Step 1 — introduce the event type; do not use it yet.**
`agent/trigger/events.py`: a frozen `RunEvent` dataclass and the event-name
constants. No call sites. Pure addition.

**Step 2 — make `run()` a drain over a generator.**
Extract the graph execution into `ReActLoop.iter_run()` yielding events;
`run()` becomes `for event in self.iter_run(...): pass` plus the existing
result assembly. **The whole existing suite must pass untouched** — that is
the acceptance criterion for this step and the reason it comes before any
endpoint change.

**Step 3 — lift the I/O out of the nodes; nodes become generators.**
Still no behaviour change: `run()` keeps its sync driver and the suite keeps
passing untouched.

This step exists because "share the node implementations" between a sync and
an async driver is not something you can hand-wave. `_act_node` contains a
blocking `self._dispatcher.dispatch(...)`; an async driver needs
`await asyncio.to_thread(...)` there, and a `def` cannot `await`. The only
two ways out are duplicating `_act_node` — which is the disease — or taking
the I/O out of the node.

Everything in a node except two calls is pure: budget checks, loop
detection, the reflection flag, step recording, the routing decision. So the
nodes declare the I/O they need and the driver performs it:

```python
def _think_node(self, state):
    ...budget / loop / reflect decisions, all pure...
    response = yield CallModel(managed, offered)     # driver sends the result back
    ...record the step, add tokens, set __next__, still pure...
    return state

def _act_node(self, state):
    for tc in tool_calls:
        observation = yield CallTool(tc.name, tc.arguments)
        step.record_tool_call(...)
    return state
```

The sync driver is about ten lines:

```python
def _drive(self, node, state):
    generator = node(state)
    request = next(generator)
    while True:
        result = self._perform(request)              # chat() / dispatch()
        try:
            request = generator.send(result)
        except StopIteration as done:
            return done.value
```

Node logic is written once; the sync/async difference is confined to
`_perform`. `StateGraph` learns that a node may be a generator — that is the
only change to the graph layer besides `iter_steps()`.

Side benefit worth having on its own: a node driven by a list of canned I/O
results is testable with no `MockLLM` subclass and no fake registry at all.

**Step 3.5 — add the async driver.** Symptom 1 fixed.
`_aperform` awaits `asyncio.to_thread(self._dispatcher.dispatch, ...)` for a
`CallTool`; everything else is the sync driver with `await` in one place.
`aiter_run()` is the async counterpart of `iter_run()`.
Regression test: the asyncio-interleave repro (a 300ms tool must not delay a
30ms coroutine past ~50ms).

**Step 4 — repoint `/api/stream` at `aiter_run()`.**
`_stream_leader_steps` becomes a translator: `RunEvent` → `_sse(...)`. This
is where ~170 lines of duplicated control flow are deleted, and where
symptoms 2 and 3 disappear as a consequence rather than as separate fixes —
suspension and step recording come from the loop because there is no longer a
second implementation that could lack them.
Regression tests: a suspending tool must yield a `suspended` event *and* save
a checkpoint reachable by `POST /api/runs/{id}/resume`; a two-model-call run
must report `steps == 2` with the prompt counted in `tokens`.

**Step 5 — streaming tokens.**
`_think_node` yields `CallModel`; the async `_aperform` satisfies it with
`llm.astream()` when the provider has it, emitting `text` events as tokens
arrive, and falls back to `chat()` otherwise. Nothing in the node changes —
this is the payoff for step 3. The sync driver keeps using `chat()`, so
`/api/run` is untouched.

**Step 6 — delete the duplicate.**
Remove `_stream_leader_steps`' remaining loop body and the
`REFLECT_AFTER_FAILURE_STATE_KEY` handling in `app/server.py`. Drop the
cross-referencing comments in `react_loop.py` that exist only to warn about
the copy. Their disappearance is the real deliverable.

---

## 4. Decisions worth stating up front

**Why a generator rather than callbacks.** A callback interface would let the
loop stay a single `run()` method, but it inverts control: the endpoint could
not `break` out of a run (client disconnect) without an exception or a
sentinel return, and back-pressure on a slow SSE client would have nowhere to
live. A generator gets both for free — the consumer's pace *is* the loop's
pace.

**Why `asyncio.to_thread` rather than making tools async.** Making `BaseTool`
async would push `async` through `ToolRegistry`, `ToolDispatcher`, every
built-in tool and every user-defined one, to solve a problem only the HTTP
layer has. The MCP bridge already demonstrates the alternative (BUGS.md #6):
keep the async-ness in one place and hand the rest of the system a
synchronous API. `to_thread` is that same choice, one layer up — and after
step 3 it lives in the driver, so no node knows it happened.

**Why nodes become generators rather than each running in a thread.** The
simpler-looking alternative is to leave the nodes exactly as they are and
have the async driver run each whole *node* in `asyncio.to_thread`. That
fixes symptom 1 with no node changes at all, and it is genuinely tempting.

It dead-ends at step 5. A synchronous node running in a worker thread that
wants to consume `llm.astream()` — an async generator — has to stand up its
own event loop to do it: the MCP bridge pattern again, pointing the wrong
way. And per-token output is the entire reason `/api/stream` exists; the
frontend already renders it. Trading the endpoint's main feature for a
smaller mechanical diff up front is the wrong trade.

Generator nodes are also the *same* trick as the outer `iter_run`, applied
one level down — one concept rather than two — and they strengthen rather
than weaken the "nodes are pure `(state) -> state`" rule this repo already
commits to (AGENTS.md §5): moving the I/O out makes a node **more** pure, not
less.

**Why not a callback/effect-handler object instead of `yield`.** Passing an
`invoke` callable into each node would work for the sync driver and fail for
the async one for the same reason step 3 exists: the node would have to
`await` the callable's result. `yield` is what lets one function body be
driven by either.

**Why the sync path is not removed.** `ReActAgent.run()` is the documented
Python API, used by the eval harness, `MultiAgentOrchestrator`'s worker
threads, and every example. It should not require an event loop.

**Token accounting.** The streaming path cannot read `usage` from a provider
that does not send it. It should use the same `_prompt_tokens()` estimate the
sync path now uses for its pre-call budget check (BUGS.md #21), and mark the
result as estimated rather than reporting an estimate as measured.

---

## 5. Cost and risk

Roughly 500 lines moved, ~170 deleted, across `agent/trigger/react_loop.py`,
`agent/trigger/graph.py` (`iter_steps()`, plus generator-aware nodes), a new
`agent/trigger/events.py`, and `app/server.py`.

Steps 1, 2, 3 and 3.5 are behaviour-preserving by construction, and each has
the same acceptance criterion: **the existing suite passes with no test
changed**. Step 3 is the largest of them — it rewrites the I/O boundary of
both nodes — but it cannot change what a run does, because the sync driver
performs exactly the calls the node used to make inline.

The real risk is step 4: `/api/stream` is what the playground UI uses, so a
regression there is immediately visible. That is why the ordering matters
more than the total size.

Two failure modes to avoid:

- **Stopping after step 2.** The loop gets a generator API and the endpoint
  keeps its own copy — a third thing to keep in sync, strictly worse than
  today.
- **Skipping step 3 and duplicating `_act_node` for the async driver.** It is
  the shortest path to a working `aiter_run()` and it re-creates the exact
  defect this whole document exists to remove, one layer lower down.
