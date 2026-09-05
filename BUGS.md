# Bugs

A record of real bugs found in this codebase — root cause, how each was
found, the fix, and the regression test guarding it. Kept separate from the
README's Roadmap (which tracks missing *features*, not defects) so a defect
already diagnosed here doesn't get silently rediscovered later.

## Format

Each entry:

- **Found**: date + how it surfaced (a real error, a manual repro, a review)
- **Symptom**: what was actually observed
- **Root cause**: the actual mechanism, not just the trigger
- **Fix**: what changed, file:line, commit
- **Regression test**: the test that now guards it
- **Status**: `Fixed` / `Open` / `Won't fix` (with why, if the latter)

---

## 1. `ShortTermMemory` windowing could split a `tool_calls` message from its own tool responses

- **Found**: 2026-08-30, from a real `openai.BadRequestError` reported while
  delegating a subagent through the running server:
  `Messages with role 'tool' must be a response to a preceding message with
  'tool_calls'`.
- **Symptom**: mid-conversation, delegating a subagent (or any multi
  tool-call step) could 400 out of nowhere on a later LLM call, with no
  obvious relation to what was actually being asked at that moment.
- **Root cause**: `ShortTermMemory.manage()` (`agent/state/memory.py`)
  trimmed old messages with a naive tail-slice by message count — keep the
  last `window` messages, summarize everything before that into one system
  message. It had no concept of "this assistant message and the N tool
  messages right after it are one atomic unit." When a delegation step
  (assistant + parallel `spawn_subagent` tool_calls + their tool results)
  happened to straddle that cut line, the assistant message got folded into
  the summarized half while its tool responses survived into the kept half
  — leaving "orphaned" `tool` messages with no `tool_calls` message in front
  of them. Every OpenAI-compatible chat API rejects that shape outright.
  Reproduced directly against `ShortTermMemory.manage()` with 5 prior turns
  + a 2-call delegation step + 5 more turns (24 messages, `window=12`) —
  reliably cut right between the assistant's `tool_calls` message and its
  two tool responses.
  Pre-existing in `agent/state/memory.py`, used identically by `ReActLoop`
  (`/api/run`) and the SSE reimplementation (`/api/stream`) — not introduced
  by the conversation-continuity work, but made easier to hit by it:
  replayed prior-turn history raises a run's starting message count before
  a new task even begins, and multi-agent delegation produces several
  messages per step — both push the 12-message default window closer
  sooner.
- **Fix**: partition messages into atomic units first (`_tool_call_groups`,
  `agent/state/memory.py`) — an assistant message with `tool_calls` plus
  every `tool` message immediately following it is one unit, everything
  else is its own unit — then build the kept window from whole units off
  the tail instead of a raw message-count slice. The kept window can now
  run slightly over `window` messages when the boundary unit is itself
  larger than one message; that's the correct trade-off over shipping a
  malformed request. Commit `c7c218a`.
- **Regression test**:
  `test_short_term_memory_never_separates_a_tool_calls_message_from_its_tool_responses`
  (`tests/test_agent.py`) — reproduces the exact reported shape and asserts
  no `tool` message in the managed output lacks a preceding assistant
  message declaring its `tool_call_id`.
- **Status**: Fixed.

---

## 2. `/api/stream`'s `conversation_id` resolved to an unusable `Query` object instead of `None` when called directly

- **Found**: 2026-08-30, while adding tests for the new `conversation_id`
  parameter on `/api/stream` — an existing test broke with
  `TypeError: Object of type Query is not JSON serializable`.
- **Symptom**: any test (or other code) calling the `stream()` route
  function directly — bypassing FastAPI's request-parsing layer, which is
  how every server test in this repo is written (see `AGENTS.md` §9) —
  without passing `conversation_id` explicitly would get the unresolved
  `fastapi.Query(...)` sentinel object as the value, not the declared
  default of `None`.
- **Root cause**: `conversation_id: Optional[str] = Query(None, ...)` only
  resolves its default through FastAPI's dependency-injection layer when a
  real HTTP request comes in. A bare Python call to the function — exactly
  what this project's own test convention does — never goes through that
  layer, so the parameter keeps whatever object was written as the type
  hint's default: the `Query(...)` field-info object itself, which is
  truthy and not a string, breaking every downstream `if conversation_id:`
  check and any attempt to serialize it.
  Real HTTP traffic through the actual ASGI app was never affected — only
  this repo's direct-call testing style.
- **Fix**: use a plain `conversation_id: Optional[str] = None` instead of
  wrapping it in `Query(...)` (`app/server.py`) — functionally identical for
  real requests (still an optional query string parameter), and directly
  callable in tests without dependency resolution. Commit `e76d12a`.
- **Regression test**: the whole `/api/stream` test block in
  `tests/test_server_conversation.py` exercises the route function directly
  with `conversation_id` omitted; any regression to a `Query(...)`-wrapped
  default fails immediately with the same `TypeError` this bug originally
  produced.
- **Status**: Fixed.

---

## 3. Windows `pytest` failed outright on `tmp_path` fixtures

- **Found**: 2026-08-30, running the existing test suite before adding new
  tests — `PermissionError` on
  `C:\Users\Li Zhuoyang\AppData\Local\Temp\pytest-of-Li Zhuoyang`.
- **Symptom**: any test using the `tmp_path`/`tmp_path_factory` fixtures
  (e.g. `test_sqlite_repository_persists_structured_lifecycle`) failed on
  this machine before any test code even ran, purely from pytest trying to
  create its own basetemp directory.
- **Root cause**: environment/permissions on this Windows user profile's
  default temp directory, not a code defect — but it silently blocked an
  unrelated slice of the test suite from ever running, which is a real risk
  in its own right (a broken test looks the same as a test nobody ran).
- **Fix**: `pytest.ini` — `addopts = --basetemp=.pytest-tmp`, routing
  pytest's scratch directory into the repo checkout (gitignored) instead of
  the OS temp dir. Commit `d2faaf1`.
- **Status**: Fixed (workaround, not a code change — flagging here mainly
  so a future "why does pytest fail immediately on a fresh Windows clone"
  question doesn't need re-diagnosing).

---

## 4. Every provider call inherited the SDK's 600-second timeout and two invisible retries

- **Found**: 2026-09-05, during a review pass over the production request
  path. Not from a failure in the wild — which is precisely the problem:
  the failure mode this creates is *slow*, not loud, so it would have shown
  up first as "the agent sometimes just hangs" with nothing in the logs.
- **Symptom**: three distinct behaviours, all invisible from inside the
  agent:
  1. **A hung provider held one ReAct step for ten minutes.** Nothing in
     the loop could catch it: `max_steps` counts steps and `max_tokens`
     counts tokens, so neither guardrail has any concept of wall-clock
     time. A `/api/run` caller saw a request that neither answered nor
     failed.
  2. **A permanently-broken request was retried anyway.** The `400` from
     BUGS.md #1 (`Messages with role 'tool' must be a response to a
     preceding message with 'tool_calls'`) is deterministic — the same
     malformed transcript fails identically every time — but the SDK sent
     it three times before surfacing it, turning one fast failure into
     three slow ones.
  3. **A run that only succeeded on the SDK's second internal attempt was
     indistinguishable from one that succeeded first try.** Nothing was
     logged between `llm.call.started` and `llm.call.completed`, so a
     provider degrading steadily produced no signal at all until it
     degraded past the point of working.
- **Root cause**: none of the three client constructors
  (`OpenAILLM.__init__`, `DeepSeekLLM.__init__`, `BailianLLM.__init__` in
  `agent/llm.py`, plus `OpenAICompatibleEmbeddingProvider.__init__` in
  `agent/memory/embeddings.py`) passed `timeout` or `max_retries` to
  `OpenAI(...)`. The `openai` SDK defaults are `timeout=600.0` and
  `max_retries=2`. Both defaults are reasonable for a script and wrong for
  an agent loop: an agent makes many calls per request and multiplies any
  per-call latency by its step count, and it owns a retry decision the SDK
  cannot make (the SDK cannot tell a malformed-transcript `400` — a bug to
  fix — from a `429` — a condition to wait out).
  A fourth issue was latent in any naive fix: adding a retry loop *on top*
  of the SDK's own multiplies rather than replaces it (3 attempts × 2 SDK
  retries = 9 real HTTP requests for one logical call), and the SDK's
  attempts stay invisible to the outer loop's logging and deadline.
- **Fix**: new `agent/retry.py` — a `RetryPolicy` dataclass (per-attempt
  `timeout_seconds`, `max_attempts`, exponential backoff with jitter, and a
  `total_deadline_seconds` ceiling across all attempts) plus
  `call_with_retry()`, which classifies failures instead of retrying
  blindly:
  - retryable (`408/409/425/429/5xx`, and connection/timeout exception
    classes that carry no status at all) → backoff and retry;
  - non-retryable (`400/401/403/404/405/413/422`) → raise
    `PermanentLLMError` immediately, one HTTP request spent;
  - unrecognized → treated as **not** retryable, matching the dispatcher's
    fail-closed stance in `agent/trigger/dispatch.py`.
  `client_kwargs(policy)` returns `{"timeout": ..., "max_retries": 0}` and
  is applied at all four construction sites, so the SDK's own retries are
  disabled and this module's policy is the only one in play.
  Errors are a two-tier taxonomy mirroring `agent/errors.py`'s:
  `TransientLLMError` (provider unavailable, budget exhausted — the request
  was fine) vs `PermanentLLMError` (the request itself was rejected).
  `ReActLoop._think_node` now catches `TransientLLMError` and ends the run
  with `stop_reason="llm_unavailable"` instead of letting it propagate out
  of `run()` — an escaping exception discarded the whole trajectory, every
  tool observation already collected, and the token accounting.
  `PermanentLLMError` deliberately still propagates: a request the provider
  rejects outright is a defect to fix, and degrading around it would hide
  it behind a plausible-looking partial answer.
  Tuned in `config/agent.yaml`'s new `retry:` section (defaults: 60s per
  attempt, 3 attempts, 180s total).
  Streaming (`DeepSeekLLM.astream`) gets the timeout but deliberately not
  the retries — a stream that fails partway has already yielded tokens the
  caller forwarded to a browser, so a second attempt would re-emit them as
  duplicates. Documented in the method's docstring rather than left implied.
- **Regression test**: `tests/test_retry.py` (14 tests) — notably
  `test_a_rejected_request_is_never_retried` (a `400` must cost exactly one
  HTTP request), `test_client_kwargs_disables_the_sdks_own_retries` (guards
  the multiplicative-retry trap), and
  `test_the_total_deadline_stops_retrying_even_with_attempts_left`. Plus
  `test_an_unavailable_provider_ends_the_run_without_discarding_the_trajectory`
  and `test_a_rejected_request_still_propagates_as_a_bug_to_fix` in
  `tests/test_agent.py`.
- **Status**: Fixed.

---

## 5. `AgentGateway` was never wired into the server, and its request queue only ever grew

- **Found**: 2026-09-05, while wiring the gateway into `app/server.py` for
  the first time. The queue defects were latent until then — the class had
  unit tests, but nothing in the production path had ever constructed one.
- **Symptom**: two separate problems.
  1. **The gateway did nothing.** `agent/trigger/gateway.py` implemented
     rate limiting, a concurrency cap, request queuing, and per-request
     trace ids; README's Roadmap listed it as done. But `grep AgentGateway
     app/` returned nothing: `/api/run` called
     `await asyncio.to_thread(execute)` directly and `/api/stream` had no
     admission control at all. The server had no rate limit and no bound on
     concurrent agent runs — N simultaneous requests meant N Leaders, each
     free to spawn up to `run_budget.max_subagents` Workers.
  2. **`RequestQueue` leaked one entry per request.** `AgentGateway.run()`
     called `self._queue.enqueue(trace_id)` on every request but only ever
     called `dequeue()` on the *timeout* path. An admitted request was
     never removed, so `_queue` grew without bound for the process's
     lifetime and `size` was meaningless as a saturation signal. Worse, the
     one path that did remove an entry called `dequeue()`, which pops the
     *oldest* item — not the request that actually timed out.
- **Root cause**: (1) the gateway was built as a library component and the
  wiring step was never done — the Roadmap checkbox tracked the class
  existing, not the request path using it. (2) `enqueue`/`dequeue` were
  modelled as a FIFO work queue, but the queue's actual job here is to
  *track who is currently waiting*, which is a set-like membership problem,
  not a FIFO. With FIFO semantics there is no correct place to call
  `dequeue()`, which is why the success path simply didn't.
- **Fix**:
  - `RequestQueue.dequeue()` → `RequestQueue.release(req)`, removing the
    exact handle `enqueue()` returned; `_QueuedRequest` is now
    `@dataclass(eq=False)` so `list.remove` matches on identity rather than
    on field equality. `AgentGateway.admit()` pairs `enqueue`/`release` in
    a `try/finally` so both the admitted and the timed-out path release.
  - New `AgentGateway.admit()` context manager is now the real primitive
    (rate limit → queue → concurrency slot → yield `Admission`);
    `run()` became a thin wrapper over it. This exists because the server's
    unit of work is a Leader run returning `MultiAgentRunResult`, which
    carries subagent results that `GatewayResult` has no field for —
    forcing it through `run()` would have silently dropped them.
  - `app/server.py`: module-level `GATEWAY` built from a new
    `gateway:` section in `config/agent.yaml` (`enabled: false` restores
    the old admit-everything behaviour), an `_admission()` helper both
    endpoints wrap themselves in, and `_gateway_http_error()` mapping
    `RateLimitExceeded` → **429** and `QueueTimeout` → **503**, both with
    `Retry-After`. `/api/stream` claims its slot *before* the response
    starts (so a rejection is a real HTTP error, not an SSE stream whose
    first event is an error) and holds it until the generator finishes.
    `/api/health` now reports `free_slots`/`queued`.
  - Admission is acquired inside the worker thread, since
    `ConcurrencyGuard.acquire` blocks; the trade-off (a queued request
    occupies a thread while it waits) is documented on `admit()` rather
    than left for a reader to discover.
- **Regression test**: `tests/test_gateway.py` — notably
  `test_admitted_requests_are_removed_from_the_queue`,
  `test_releasing_removes_that_request_and_not_merely_the_oldest`,
  `test_a_request_that_cannot_get_a_slot_in_time_is_rejected_and_not_leaked`,
  and `test_run_endpoint_returns_429_when_the_gateway_rate_limits` (which
  fails outright if the gateway is ever unwired again).
- **Status**: Fixed.

---

## 6. All MCP tool calls were serialized process-wide, and queued calls reported false timeouts

- **Found**: 2026-09-05, reading `MCPManager._worker` while reviewing the
  multi-agent latency numbers. The published `benchmarks/multi_agent_latency`
  speedups (2.33x/4.00x/6.50x at k=2/3/6) did not exercise this path — those
  Workers used local tools only — so a real fan-out over MCP tools would not
  have matched them.
- **Symptom**: two Workers running concurrently and each calling an MCP tool
  took turns instead of overlapping: total latency was the *sum* of their
  call durations, not the max. And the second call frequently failed with
  `MCP tool fetch__fetch timed out.` even when the server was healthy and
  the call itself was fast.
- **Root cause**: `MCPManager` runs one background thread owning one asyncio
  event loop, fed by a queue — a sound design (it keeps the async-ness
  inside the module, so `ToolRegistry`/`ToolDispatcher`/`ReActLoop` never
  had to become async). But `_worker` handled a `"call"` item by *awaiting*
  it inline inside the `while True` loop:

  ```python
  kind, payload, future = await self._queue.get()
  ...
  result = await asyncio.wait_for(session.call_tool(...), timeout=...)
  ```

  so the next queue item could not even be read until the current call
  finished. One slow call blocked every other call in the process,
  regardless of which server it targeted.
  The false timeout followed directly: the caller-side wait was
  `future.result(timeout=config.call_timeout + 1.0)`, a budget sized for
  *executing* the call, while the future's real wall-clock life also
  included however long it sat behind other calls. A call that had never
  been sent hit its caller-side timeout and was reported as
  `"MCP tool ... timed out"` — indistinguishable in the logs from a
  genuinely slow tool, which is what made the underlying serialization hard
  to see.
- **Fix**: `agent/mcp/client.py` — `_worker` now dispatches each call as its
  own task (`asyncio.create_task(self._run_call(...))`) and tracks it in an
  `in_flight` set, so the loop returns to `queue.get()` immediately and
  calls genuinely overlap. Per AGENTS.md §8, the new unbounded resource got
  an explicit limit in the same change: `MCPServerConfig.max_concurrent_calls`
  (default 4, per server rather than global — a local stdio process and a
  hosted HTTP endpoint tolerate very different loads) enforced by an
  `asyncio.Semaphore`, and `MCPServerConfig.slot_wait_timeout` (default 30s)
  bounding the wait for one.
  The two waits are now reported as the different things they are: exceeding
  `call_timeout` says the call "exceeded its Ns call timeout", while
  exceeding `slot_wait_timeout` says it "waited Ns for a free slot ... and
  was never sent". The caller-side backstop was corrected to
  `slot_wait_timeout + call_timeout + 5.0` — the sum of the deadlines the
  worker actually enforces — and is now only a guard against a dead worker
  thread, not a deadline in its own right.
  `close` cancels in-flight tasks and gathers them before tearing the
  sessions down, so shutdown can't race a call still holding a session.
  **No architectural change was needed**: the event loop, the queue, and the
  thread bridge were already right; the bug was awaiting in the wrong place.
  The synchronous `MCPManager.call()` signature is unchanged, so nothing
  above this module was touched.
- **Regression test**: `tests/test_mcp.py` —
  `test_concurrent_calls_overlap_instead_of_taking_turns` (4 × 0.3s calls
  must finish in well under the 1.2s a serialized run would take, and the
  fake client must observe 4 overlapping),
  `test_per_server_concurrency_is_bounded_by_max_concurrent_calls`, and
  `test_a_call_that_never_gets_a_slot_says_so_rather_than_claiming_a_timeout`.
- **Status**: Fixed.

---

## 7. A deliberately cancelled job was reclassified as a fatal tool error

- **Found**: 2026-09-05, reading the log output of a passing test
  (`test_cancellation_actually_reaches_inside_the_tool`). The test was
  green; the captured log said `job.failed` with a `FatalToolError`
  traceback. The assertion and the logs disagreed, and the logs were right
  about something the assertion wasn't checking.
- **Symptom**: cancelling a long-running job logged it as a failure with a
  full traceback, and `JobRunner._execute`'s `except JobCancelled` branch
  never ran — the job only ended up in the right state by accident,
  because `JobRunner.cancel` had already marked it `CANCELLED` and
  `_finish` refuses to overwrite a terminal verdict.
- **Root cause**: `JobCancelled` was a plain `Exception`. It is raised
  *inside* the tool by `JobContext.raise_if_cancelled()`, so on the way
  out it passed through `FunctionTool.run`, whose whole job is to classify
  any unrecognized exception into the two-tier tool taxonomy — turning it
  into a `FatalToolError` before the runner ever saw it. Cooperative
  cancellation is control flow, not a tool failure, but nothing in the type
  hierarchy said so, so the classifier did the only thing it could.
- **Fix**: added `ControlSignal` to `agent/errors.py` — a hierarchy
  deliberately disjoint from `ToolCallError`, for a tool telling the loop
  to do something other than continue when nothing has gone wrong.
  `FunctionTool.run` and `ToolDispatcher.dispatch` now re-raise it
  untouched (the dispatcher logs `tool.call.signalled` rather than
  `tool.call.failed`). `JobCancelled` and `SuspendRun` both subclass it.
  Without this, `SuspendRun` — the signal the entire long-running-job
  design rests on — would have hit exactly the same wall: a suspension
  request reclassified into a fatal error one layer before the loop could
  act on it.
  AGENTS.md §6 gained the rule, since this is a new hierarchy rather than
  an application of an existing one.
- **Regression test**:
  `test_a_cancelled_job_is_not_reported_as_a_failure` (`tests/test_jobs.py`)
  asserts on the captured log rather than only on the final status, since
  the status was already correct for the wrong reason.
- **Status**: Fixed.

---

## 8. `LocalCommandTool`'s timeout kills only the direct child, orphaning its descendants

- **Found**: 2026-09-05, from a design review of what it would take to
  support long-running tools, then confirmed with a standalone repro on
  this Windows machine (see below). Not observed in production traffic, but
  the default configuration is squarely exposed: the CLI tool is opt-in
  (`ENABLE_LOCAL_CLI=1`), yet once on, the default allowlist is
  `git,rg,python,python.exe,pytest` (`app/server.py:177`) — `pytest` and
  `python` spawn children as a matter of course (this repo's own suite
  launches MCP stdio servers), and `git` spawns a pager/credential helper.
  Only `rg` is reliably a leaf. Default timeout is 30s
  (`AGENT_CLI_TIMEOUT_SECONDS`) and the default workspace root is the repo
  itself, so `run_command(["pytest"])` against this very checkout is both
  the most likely first use and a direct trigger.
- **Symptom**: when an allowlisted command exceeds its timeout, the tool
  returns `Command timed out after Ns` and the run continues cleanly — but
  any process the command itself spawned keeps running, unattached and
  unreachable, until it finishes on its own or the machine reboots. For a
  command tree like `npm` → `node` → workers, or `git` → pager/credential
  helper, or `python script.py` → a worker pool, the reported timeout is a
  lie: the expensive part is still burning CPU.
- **Root cause**: `LocalCommandTool.run()` (`agent/local_tools/tools.py:260`)
  launches with a plain `subprocess.Popen` — no `start_new_session=True`
  (POSIX) and no `CREATE_NEW_PROCESS_GROUP`/Job Object (Windows) — so the
  spawned process shares this process's group and gets no kill-group label
  of its own. The timeout handler then calls `process.kill()`
  (`tools.py:290`), which signals exactly one PID. Descendants are not
  reachable from that call: killing the parent immediately reparents them
  (to `init`/PID 1 on POSIX; on Windows the PPID field dangles), so even a
  after-the-fact `children(recursive=True)` walk would find nothing —
  the parent-pointer chain is the only index into the tree, and killing
  the parent is precisely what destroys it. A process tree is a linked
  list that breaks; killing a group needs a label the kernel maintains on
  every member (process group / Job Object / cgroup), applied at spawn
  time, not a tree walked at kill time.
  Secondary effect: the two `drain` reader threads (`tools.py:281-286`)
  read from pipes whose write ends the surviving grandchildren still hold,
  so `stream.read()` never sees EOF. `reader.join(timeout=1)` stops that
  from hanging the run, but the threads and their pipe handles leak.
- **Repro** (dependency-free, run from anywhere; mirrors `tools.py:260-294`
  exactly): `Popen` a `python -c` that itself `Popen`s a `time.sleep(300)`
  child and prints the grandchild's pid, let it exceed a 1s timeout, call
  `process.kill()`, then check the grandchild with `tasklist /FI "PID eq N"`.
  Observed on this machine 2026-09-05:
  ```
  child alive after kill      = False
  GRANDCHILD alive after kill = True
  ```
  Swapping the single `process.kill()` for
  `taskkill /PID <pid> /T /F` on the same script gives
  `GRANDCHILD alive after kill = False`.
- **Fix**: not applied yet. Intended change, both verified in the repro
  above: spawn with a kill-group label —
  `start_new_session=True` on POSIX, `creationflags=CREATE_NEW_PROCESS_GROUP`
  on Windows — and in the `TimeoutExpired` handler kill the group rather
  than the pid: `os.killpg(os.getpgid(process.pid), SIGKILL)` on POSIX,
  `taskkill /PID <pid> /T /F` on Windows. Note `taskkill /T` still walks
  the tree, so it inherits the same race for processes spawned between
  enumeration and kill; it works here because it runs while the parent is
  still alive. A Windows Job Object with
  `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` is the airtight version and is the
  right move if this tool ever runs untrusted or deeply-nested commands.
- **Regression test**: none yet. The repro above is the shape to port —
  assert the grandchild pid is gone after the timeout path runs.
- **Status**: Open. The same defect class blocks the planned long-running
  tool work (a `job_id`-addressed runner has to be able to actually kill
  what it started), so this is worth fixing as part of that rather than in
  isolation.

---

# External review, 2026-09-05

Entries #9-#22 come from one external code review (an independent model was
pointed at the working tree and asked to find defects); #23 and #24 were
found while fixing #22. Every claim in
that report was re-derived here before being acted on — each has a standalone
repro that ran against the pre-fix code, and the repros are what the
regression tests were written from. Nothing was taken on the report's word.

Two things worth recording about the review itself. It grouped what are
really *three symptoms of one root cause* (the SSE endpoint being a
hand-written copy of `ReActLoop`, now #22) as three separate findings, so the
headline count was higher than the number of independent defects. And two of its
findings were correctly *observed* but wrongly *framed* — see #21, which is a
contract that needed stating rather than behaviour that needed changing, and
#20, where the skip logic was defensible and the return value was not.

---

## 9. Every container parameter was advertised to the model as its element type

- **Found**: 2026-09-05, external review; confirmed by calling `_json_type`
  directly on `list[int]` and `dict[str, int]`.
- **Symptom**: a tool declaring `names: list[str]` published a schema saying
  `{"type": "string"}`. A model following that schema sends `"alice"` where
  the tool binds a list, which surfaces as a `TypeError` from inside the
  tool — routed to the dispatcher's `invalid_arguments` retry path, so the
  model is told to "check the arguments and retry" against the same wrong
  schema. It retries, fails identically, and burns its retry budget. The
  measured mapping was `list[int] -> integer`, `dict[str, int] -> string`,
  `List[str] -> string`.
- **Root cause**: `_json_type` (`agent/tools.py`) had one branch for
  *anything* with a `get_origin()`, and that branch implemented `Optional[X]`
  / `Union[...]` semantics — "use the first non-`None` type argument". Every
  generic collection has an origin, so every generic collection was treated
  as a Union of its type parameters. `Optional[int] -> integer` is correct;
  `list[int] -> integer` is the same line of code being wrong. The bug was
  invisible in this repo's own tools because none of them take a container
  parameter — it only bites someone extending the kit, which is exactly the
  audience `@tool`'s auto-schema exists for.
- **Fix**: `_json_type` → `_json_schema`, returning a full schema fragment
  rather than a bare type string, with the cases separated:
  Union/Optional (both `typing.Union` and the 3.10+ `X | None` spelling,
  which has a *different* origin) → first non-`None` member; sequences →
  `{"type": "array", "items": ...}`; mappings → `{"type": "object",
  "additionalProperties": ...}`; `Literal[...]` → an `enum`, which is the
  single most useful thing a tool schema can tell a model. A heterogeneous
  `tuple[int, str]` gets an untyped array rather than a confident lie.
  Recursive, so `Mapping[str, list[int]]` describes both levels.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_a_container_parameter_keeps_its_container_type`,
  `test_optional_still_describes_the_wrapped_type` (guards the case that was
  already right), `test_a_heterogeneous_tuple_is_not_described_as_a_typed_array`,
  `test_a_tools_generated_schema_reflects_its_annotations`.
- **Status**: Fixed.

---

## 10. Malformed tool-call JSON became an empty call that tools happily executed

- **Found**: 2026-09-05, external review; confirmed by driving
  `OpenAILLM.chat` with a fake client returning a truncated `arguments`
  string.
- **Symptom**: a provider that truncated or garbled a tool call's arguments
  produced a call with **no** arguments. For a tool whose parameters all
  have defaults, that is a perfectly valid call, so it *ran* — with defaults
  the model never chose, returning a plausible observation. Measured:
  `'{"text": "target-A'` → `{}` → `send()` executed and returned
  `"sent to default-target"`. Nothing in the transcript told the model its
  call had been malformed, so it had no reason to correct anything.
- **Root cause**: `except json.JSONDecodeError: args = {}` in all three
  places tool calls are decoded (`OpenAILLM.chat`, `DeepSeekLLM.chat`,
  `DeepSeekLLM.astream` in `agent/llm.py`; `BailianLLM` inherits the first).
  `{}` was chosen as a safe default, and it would be — for a tool with
  required parameters, binding fails and the dispatcher's existing retry
  path handles it. The defect only appears when every parameter has a
  default, which is precisely when the empty call is indistinguishable from
  a real one. A "safe" fallback that is silently *valid* is worse than an
  error.
- **Fix**: new `agent.llm.parse_tool_arguments()`, used by all three sites.
  It returns `{}` only for a genuinely empty payload (a real no-argument
  call) and otherwise returns the **raw text** when it will not decode.
  A non-dict then lands in `ToolDispatcher.dispatch`'s existing, already
  tested `not isinstance(arguments, dict)` branch — a recoverable error that
  tells the model and gives it a retry. The message now includes the text
  that failed to parse, since "must be a JSON object" is unhelpful when the
  model believes it sent one. `ToolCall.arguments` is typed `Any` with the
  reason documented on the field.
  Deliberately no new error class: the right handling already existed, the
  parse failure just was not being routed to it.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_unparseable_tool_arguments_are_not_flattened_to_an_empty_call` and
  `test_a_tool_with_all_defaults_is_not_run_on_malformed_arguments`, which
  asserts the tool body never ran (the assertion the original code would
  have passed without).
- **Status**: Fixed.

---

## 11. Job deduplication treated "Alpha" and "alpha" as the same work

- **Found**: 2026-09-05, external review; confirmed by comparing
  `fingerprint()` output for two case-differing argument dicts.
- **Symptom**: submitting a long-running tool with `text="Alpha"` and then
  `text="alpha"` returned the *same* `job_id`, so the second submission was
  handed the first one's result. Case-sensitive URL paths, query strings,
  file contents and identifiers were all affected.
- **Root cause**: `fingerprint()` (`agent/jobs/runner.py`) lowercased the
  entire canonicalized argument JSON. Its own docstring said why: it was
  "canonicalized the same way loop detection canonicalizes a call", copied
  from `_detect_loop` in `agent/trigger/react_loop.py`.
  That is the actual defect — not the `.lower()` itself, but reusing a
  *heuristic's* canonicalization as an *idempotency key*. `_detect_loop`
  over-matching costs at worst one early stop, so lowercasing there is a
  reasonable way to catch equivalent spellings. This key decides whether to
  return a cached result instead of doing the work, so over-matching returns
  the wrong answer. Same code, different contract.
- **Fix**: dropped `.lower()`; `sort_keys=True` stays, so key *order* is
  still normalized and value case is not. The docstring now says explicitly
  that it deliberately no longer matches `_detect_loop`, and why — otherwise
  the next person to notice the divergence "fixes" it back.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_the_job_fingerprint_is_case_sensitive`,
  `test_the_job_fingerprint_still_ignores_key_order` (the property that had
  to survive), `test_case_differing_submissions_are_two_jobs` end-to-end
  through `JobRunner.submit`.
- **Status**: Fixed.

---

## 12. A heartbeat could resurrect a job that had already been cancelled

- **Found**: 2026-09-05, external review; confirmed with a hand-interleaved
  repro, then re-confirmed through the real `JobRunner._heartbeat`.
- **Symptom**: `cancel(job_id)` returned a job marked `CANCELLED`, and a
  progress heartbeat arriving immediately afterwards moved it back to
  `RUNNING`. The caller had already been told the job was cancelled. The
  same shape applied to the duration watchdog and to `_execute`'s initial
  `RUNNING` transition — a job cancelled between `submit()` and the worker
  thread picking it up was marked running again.
- **Root cause**: every state transition in `JobRunner` was
  `store.get()` → mutate the snapshot → `store.put()`, with an
  `if job.terminal: return` guard between the get and the put. That guard
  reads a value another thread is free to change before the put lands: a
  textbook check-then-act race. `JobStore` offered only unconditional `put`,
  so there was no way to express "write this **only if** the stored row is
  still non-terminal" — the check had to live in the caller, where it cannot
  be atomic. `_finish` already had a comment explaining that a terminal
  verdict must win; the comment was right and the mechanism could not
  enforce it.
- **Fix**: moved the condition into the store, where it can be atomic.
  `JobStore` gained two primitives, implemented in both `InMemoryJobStore`
  (under its existing `RLock`) and `SQLiteJobStore` (one statement with a
  `status NOT IN (terminal)` predicate):
  - `put_if_not_terminal(job) -> bool` — compare and write together;
  - `heartbeat(job_id, at=, progress=) -> bool` — a *partial* update, so a
    heartbeat cannot carry a stale copy of any other field back into the
    store even in principle.
  `_heartbeat`, `_finish`, `_time_out`, `cancel` and `_execute` all use
  them. `cancel` additionally re-reads and returns what is actually stored
  when it loses the race, rather than the object it optimistically built.
- **Regression test**: `tests/test_review_fixes.py`, parametrized over both
  store implementations — `test_a_heartbeat_cannot_resurrect_a_cancelled_job`,
  `test_a_job_cancelled_before_its_worker_starts_stays_cancelled`,
  `test_put_if_not_terminal_refuses_to_overwrite_a_terminal_verdict`, plus
  the positive cases so the guard cannot be "fixed" by making it always
  refuse.
- **Status**: Fixed.

---

## 13. Reading a memory could undo its deletion

- **Found**: 2026-09-05, external review; confirmed against
  `InMemoryMemoryRepository` and re-confirmed through `MemoryManager.recall`.
- **Symptom**: a memory deleted (tombstoned, expired, or superseded) while a
  concurrent `recall()` was in flight came back `ACTIVE`, with `deleted_at`
  cleared. Once the vector index was rebuilt, the deleted content was
  searchable again. For a subsystem whose entire purpose is policy-controlled
  retention, a delete that a *read* can undo is the worst available failure.
- **Root cause**: `MemoryManager.recall()` stamps `last_accessed_at` on every
  hit, and did it as `repository.get()` → mutate → `repository.update()`.
  `update()` replaces the whole record, so the write carried the entire
  pre-deletion snapshot — status, `deleted_at`, `deletion_reason` and all —
  back over the delete. The `record.status != ACTIVE` check ran against the
  stale read, so it could not see the delete either.
  Secondary, and independent of any race: this rewrote every field of every
  hit to change one timestamp, so a ten-result recall cost ten full-row
  writes.
- **Fix**: `MemoryRepository.touch_if_active(record_id, accessed_at=)`,
  implemented in both repositories — a conditional partial update that writes
  only the access timestamps and only while the record is still `ACTIVE`. The
  SQLite version carries `status=?` in the `WHERE` clause, so a delete that
  commits first leaves the UPDATE matching zero rows.
  It returns the updated record or `None`, and `recall()` drops any hit that
  returns `None` — which also correctly handles the ordinary (non-racing)
  case of the vector index lagging behind a deletion, since the index is
  derived data and is allowed to lag while the repository is not.
- **Regression test**: `tests/test_review_fixes.py`, parametrized over both
  repositories — `test_touching_a_deleted_memory_does_not_revive_it`,
  `test_touching_an_active_memory_stamps_it`,
  `test_touching_a_missing_memory_returns_none`.
- **Status**: Fixed.

---

## 14. `ReActAgent`'s attributes were copies, so writing one silently did nothing

- **Found**: 2026-09-05, external review; confirmed by asking
  `AgentGateway.run()` for `max_steps=1` and watching the agent run 5 steps.
- **Symptom**: `AgentGateway.run(agent, task, max_steps=1)` ran the agent to
  its *original* budget. No error, no warning, no log line — the override was
  accepted and ignored. Measured: requested 1, executed 5 steps and 5 LLM
  calls.
- **Root cause**: `ReActAgent` is a thin facade over `ReActLoop`, and its
  `__init__` assigned `self.max_steps = max_steps` (and fourteen other
  attributes) as plain copies, with the comment "Expose attributes for
  backward compatibility". Reading them worked. Writing one changed the copy
  while `ReActLoop.run()` went on reading `self.max_steps` on the *loop*.
  `AgentGateway.run` does exactly that — save, overwrite, run, restore —
  which is a reasonable thing to do to an object whose attribute is real.
  Every one of the fifteen copied attributes had the same defect; `max_steps`
  is just the one something actually wrote to.
  Not reachable from HTTP: `/api/run` passes `max_steps` into
  `_build_leader_runtime`, which constructs a fresh agent. The Python API is
  affected, which includes anything a reader of the README would write.
- **Fix**: replaced the copies with generated properties that read *and*
  write through to `self._loop` (`agent/agent.py`). Identical read API, and
  writes now land. Generated from a `_DELEGATED` tuple rather than written
  out fifteen times, so a future attribute cannot be added to one list and
  forgotten in the other.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_writing_a_facade_attribute_reaches_the_loop` (both directions) and
  `test_the_gateways_per_request_step_override_is_applied_and_restored`,
  which asserts the override took effect *and* was restored afterwards.
- **Status**: Fixed.

---

## 15. `total_deadline_seconds` bounded sleeping, not elapsed time

- **Found**: 2026-09-05, external review; confirmed with a fake clock.
- **Symptom**: a policy with `total_deadline_seconds=10.0` accepted a result
  that arrived at **18.1s**. The deadline reliably stopped a run whose
  *backoff* would exceed it (which is what BUGS.md #4's test checked) and did
  nothing about a run whose *attempts* did.
- **Root cause**: `_remaining()` was consulted in exactly one place — inside
  `if attempt > 1:`, compared against the next backoff. Nothing bounded an
  attempt itself. Each attempt carried `timeout_seconds` (60s by default),
  configured once on the SDK client at construction time, so the real worst
  case was `max_attempts × timeout_seconds + backoff` — with the defaults,
  ~183s against a declared 180s ceiling, and unboundedly worse for a policy
  with a short deadline and a long per-attempt timeout. BUGS.md #4 claimed
  this ceiling existed; it was half-implemented.
- **Fix**: `call_with_retry`'s `fn` now takes one argument — **this attempt's
  timeout** — computed as `min(timeout_seconds, remaining)`, and the loop
  raises before making a call at all if the budget is already spent. The four
  provider call sites pass it straight through to the SDK's per-request
  `timeout=` (`agent/llm.py`, `agent/memory/embeddings.py`), since only the
  SDK can enforce a deadline on a socket it owns.
  `client_kwargs`' client-wide `timeout` stays as the default for anything
  not routed through `call_with_retry`, with its now-secondary role noted.
  Measured after the fix: per-attempt budgets `[10.0, 0.9]` against a 60s
  `timeout_seconds`, giving up at exactly 10.0s.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_each_attempt_is_capped_by_the_remaining_total_budget` (asserts both
  the clamped budgets handed to each attempt and the wall-clock ceiling) and
  `test_an_attempt_gets_the_policy_timeout_when_no_total_deadline_is_set`.
  `tests/test_retry.py`'s existing 15 tests were updated to the new `fn`
  signature.
- **Status**: Fixed.

---

## 16. A turn that called two tools recorded one, and could report a failure as a success

- **Found**: 2026-09-05, external review; confirmed with a single turn
  issuing `echo` (succeeds) and `fails` (raises `RecoverableToolError`).
- **Symptom**: the trajectory recorded `action = {"name": "echo"}` and
  `observation = "hi\nERROR calling 'fails': boom"`. The second tool's name
  and its failure were both gone as *structured* data — the error text
  survived only as a suffix on the first call's output. Downstream:
  `SubagentResult.tool_call_summary()` reported `[{"tool": "echo", "ok":
  true}]`, so a Leader reading a Worker's card saw one clean call where two
  had run and one had failed. `EvalHarness`'s `expect_tool` check and
  trajectory score read the same flattened view, and `_tools_used` (which
  feeds the tool selector's "never hide a tool already used" guard) missed
  the second tool entirely.
  The loop's own failure handling was *not* affected — forced reflection
  triggers off the observations, so it fired correctly. The defect is in the
  persisted trajectory and everything that consumes it.
- **Root cause**: `Step` modelled one action per step
  (`action`/`observation` singular), which was true when it was written and
  stopped being true once parallel tool calls were supported. `_act_node`
  bridged the gap by assigning `step.action = tool_calls[0]` and joining
  every observation into one string — a lossy projection with no way back.
- **Fix**: `Step.tool_calls`, one `{id, name, arguments, observation, ok}`
  entry per call, written by `Step.record_tool_call()` which also keeps
  `action`/`observation` in sync so existing consumers (the UI, checkpoints,
  `_detect_loop`) are untouched. `Step.from_dict` back-fills a single entry
  from the flattened view, so trajectories and checkpoints written before
  this change still read as one shape.
  Consumers moved to the per-call records: `_tools_used`,
  `SubagentResult.tool_call_summary` (which keeps the flattened path as a
  fallback for old data), `EvalHarness._used_expected_tool` and
  `_trajectory_score`. `reflection.forced` now logs `failed_tools` rather
  than implying the first call was the one that failed.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_every_tool_call_in_a_turn_is_recorded_separately`,
  `test_a_workers_tool_summary_reports_both_calls`,
  `test_the_flattened_view_is_still_populated_for_existing_consumers`,
  `test_a_pre_existing_trajectory_without_per_call_records_still_summarizes`,
  `test_a_legacy_step_backfills_one_call_record_on_restore`,
  `test_the_tool_selector_never_hides_a_second_tool_from_the_same_turn`.
- **Status**: Fixed. The SSE path still records no steps at all — that is
  #22, and is deliberately not patched here.

---

## 17. `expect_tool` was computed and then dropped from the pass/fail score

- **Found**: 2026-09-05, external review.
- **Symptom**: a task declaring `expect_tool: "calculator"` passed on
  substring match alone. An agent that answered "42" from the model's own
  weights, never calling the tool the task exists to exercise, scored
  identically to one that used it. Since the headline number in the README is
  the rule-based success rate, this inflated exactly the comparisons
  (all-tools vs. BM25 selection vs. delegation) the eval exists to make.
- **Root cause**: `_used_expected_tool` existed and was correct, but was
  wired only into `_trajectory_score` (a separate, secondary average).
  `_rule_score` checked substrings and `stop_reason == "finished"` and
  returned. The README described rule scoring as including the tool check;
  the code did not.
- **Fix**: `_rule_score` now fails a task whose `expect_tool` was not called
  (`used is not False`, so a task that declares no expected tool is
  unaffected). `_tool_names()` extracted so the tool check reads per-call
  records — see #16, without which this fix would still miss a tool called
  second in a parallel turn.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_a_task_that_skipped_its_expected_tool_does_not_pass`,
  `test_a_task_without_an_expected_tool_is_unaffected`,
  `test_the_expected_tool_is_found_in_a_multi_call_turn`.
- **Status**: Fixed. **The published eval numbers predate this** and were
  computed with the looser rule; they need re-running before being quoted
  anywhere.

---

## 18. The LLM judge scored "I cannot say whether this would pass" as a pass

- **Found**: 2026-09-05, external review.
- **Symptom**: `_judge_score` returned `True` for any reply containing the
  substring `"pass"` anywhere. A refusal to grade, a hedge, or a verdict with
  reasoning attached ("this does not pass the bar") all scored as passes.
  `"FAIL"` happened not to contain the substring, which is the only reason
  the metric produced plausible-looking numbers at all.
- **Root cause**: `"pass" in (resp.content or "").strip().lower()` — a
  containment test standing in for parsing an enumerated verdict. It also
  silently conflated "the judge said FAIL" with "the judge did not answer the
  question", which are different facts and only one of them is about the
  agent.
- **Fix**: match the first token against `{"pass", "fail"}` after stripping
  punctuation and formatting. Anything else is a *grading* failure: it logs
  `eval.judge.invalid_verdict` and scores `False`, because an ungradeable
  reply must not silently count as a success. The prompt now asks for exactly
  one word.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_the_judge_reads_a_verdict_not_a_substring`, parametrized over the
  replies that used to be misread.
- **Status**: Fixed.

---

## 19. CI claimed Python 3.9 support that `pip install` could never have delivered

- **Found**: 2026-09-05, external review; confirmed by reading the installed
  `mcp` package metadata (`Requires-Python: >=3.10`) and by checking whether
  anything transitively provides FastAPI.
- **Symptom**: two independent packaging defects, both of which mean a green
  local run says nothing about a clean environment.
  1. `.github/workflows/ci.yml` ran a matrix including `"3.9"`, and
     `requirements.txt` depended unconditionally on `mcp>=2.0,<3`, which
     declares `Requires-Python: >=3.10`. The 3.9 leg could not have got past
     `pip install -r requirements.txt`.
  2. Five test modules import `fastapi.testclient`, and FastAPI was not in
     `requirements.txt` at all. `mcp` pulls in `starlette`, `uvicorn` and
     `sse-starlette` but **not** `fastapi` (verified against the installed
     dependency metadata), and `TestClient` needs `httpx`, which is also not
     a FastAPI dependency. Locally both happened to be installed.
- **Root cause**: the optional dependencies carried
  `; python_version >= "3.10"` markers, which made 3.9 look *feature-reduced*
  (no `openai`, no `chromadb` — both genuinely optional) rather than
  unsupported. That framing hid the fact that a core dependency had the same
  floor. The FastAPI omission is the ordinary version of the same problem:
  the server was added to a repo that already had it installed.
- **Fix**: dropped 3.9 from the CI matrix and stated the 3.10 floor at the
  top of `requirements.txt`; the now-redundant `python_version` markers were
  removed rather than left to imply 3.9 is partially supported. Declared
  `fastapi`, `uvicorn` and `httpx` explicitly, with a comment naming what
  needs each.
- **Regression test**: none — a dependency-declaration defect can only be
  caught by installing into a clean environment, which is CI's job. Fixing
  the matrix is what makes CI able to catch the next one.
- **Status**: Fixed. Splitting into `core` / `server` / `dev` extras is the
  right next step and is not done; everything still installs together.

---

## 20. Re-publishing a document's earlier body was silently skipped

- **Found**: 2026-09-05, external review; confirmed against both repository
  implementations.
- **Symptom**: publish v1=alpha, then v2=beta, then v3=alpha (a rollback).
  The third call returned `skipped=True` along with the **superseded** v1
  document, and v2 remained the active version. The caller was told the
  content was already indexed — true but irrelevant — and handed a document
  that was not the effective one. A legitimate rollback did nothing, and said
  so in a way that read like success.
- **Root cause**: the dedupe check in `RAGIngestionService._ingest_text`
  matched on `(checksum, logical_id, chunker_version)` and ignored both
  `status` and `version`. In a corpus where documents are versioned and
  superseded, a document's identity is not its bytes: version, effective
  dates and provenance are all part of it. Matching on content alone answers
  "have I seen these bytes" when the question is "is this the live version".
  This is the one finding where the report's framing was off — it read as
  "the skip is wrong". The skip is a reasonable optimisation; the wrong part
  was skipping against a *non-live* row, and returning it as though it were
  current.
- **Fix**: the skip now requires the match to be `ACTIVE` **and** at the same
  `version` — i.e. a genuine re-ingest of what is currently live. Anything
  else publishes a new version normally, and the result carries a warning
  naming the version whose body it duplicates, so an operator can see a
  rollback happened rather than inferring it.
  `find_by_checksum` was made deterministic in the same change: with
  identical content now legitimately present at several versions, "any
  matching row" was ambiguous, and the two implementations disagreed
  (in-memory returned insertion order, SQLite returned `rowid DESC`). Both
  now prefer the `ACTIVE` row.
- **Regression test**: `tests/test_review_fixes.py`, parametrized over both
  repositories —
  `test_republishing_a_superseded_body_as_a_new_version_is_not_skipped` and
  `test_reingesting_the_live_version_is_still_skipped` (the behaviour that
  had to survive).
- **Status**: Fixed.

---

## 21. `max_tokens` ignored the prompt, so one call could overshoot it without limit

- **Found**: 2026-09-05, external review.
- **Symptom**: `max_tokens=1` with a call reporting 101 tokens of usage
  finished successfully, `stop_reason="finished"`, `tokens=101`.
- **Root cause**: `ctx.over_budget()` compares `tokens_used` — tokens already
  *spent* — against the ceiling, and runs at the top of `_think_node`. A run
  one token under its limit was therefore free to issue a call with an
  arbitrarily large prompt; the overshoot was bounded only by the model's
  context window.
  The report filed this as a defect. It is half a defect: a *post-hoc*
  ceiling is the correct design for an anti-runaway guardrail, which is what
  `ExecutionContext` documents this as, and no ceiling can be exact because
  nothing knows the completion size before the call. But "unbounded
  overshoot" is not a contract anyone would choose, and it was never written
  down.
- **Fix**: `_think_node` now also projects the *input* cost before calling —
  `tokens_used + _prompt_tokens(managed, offered)` — and stops the run if
  that alone would breach the ceiling. `_prompt_tokens` counts the offered
  tool schemas as well as the transcript, because a 56-tool registry is
  several thousand tokens of prompt on every step and omitting it
  systematically under-counts exactly the runs most at risk.
  This makes the overshoot bounded by the completion size rather than
  unbounded. The contract is now stated where it is enforced: a ceiling with
  a known overshoot, not a hard cap. A true cost cap needs a per-run budget
  aggregated across Workers, which is a separate piece of work.
- **Regression test**: `tests/test_review_fixes.py` —
  `test_a_call_whose_prompt_alone_blows_the_budget_is_not_made` (asserts the
  model was never called), `test_a_generous_budget_still_runs_normally`,
  `test_the_offered_tool_schemas_count_toward_the_projected_prompt`.
- **Status**: Fixed.

---

## 22. The SSE endpoint is a hand-written copy of `ReActLoop`, and has drifted

- **Found**: 2026-09-05, external review (filed there as three separate P1s);
  all three confirmed by driving `_stream_leader_steps` directly with a fake
  streaming LLM.
- **Symptom**: three defects, one cause.
  1. **The event loop blocks for the duration of every tool call.**
     `_stream_leader_steps` is an `async` generator that calls the
     synchronous `dispatcher.dispatch()` inline. Measured: with a tool
     sleeping 300ms, a coroutine scheduled to wake at 30ms woke at **342ms**.
     Every other request on the process is stalled for as long as any
     streaming run's tool takes — and MCP calls, subagent delegation and
     `run_command` are all seconds, not milliseconds.
  2. **A run that suspends on a long-running job crashes the stream.** The
     act phase catches `FatalToolError` only. `SuspendRun` is a
     `ControlSignal` (see #7) and propagates straight out of the generator.
     Measured: `uncaught_error=SuspendRun`, no `suspended` event emitted, no
     checkpoint saved. The client sees a stream that stops mid-run, and there
     is no run id to resume — the work is stranded, jobs still running.
  3. **Reported metrics are wrong.** The loop uses a local `step_idx` and
     never calls `ctx.new_step()`, but the final stats report
     `len(ctx.steps)`. Measured: 2 model calls, `done.steps = 0`. Token
     accounting is `estimate_tokens_simple(full_content)` — output text only,
     so the transcript, the tool schemas and the tool arguments are all
     uncounted. Measured: a ~1000-token prompt recorded as **5** tokens.
     `/api/stream` therefore also has no working token budget, since
     `over_budget()` reads a number that barely moves.
- **Root cause**: one cause, not three. `/api/run` executes through
  `ReActLoop`; `/api/stream` re-implements the same state machine inline in
  `app/server.py` so it can yield SSE events between phases. Everything added
  to the loop since has had to be added twice, and the copies have drifted:
  forced reflection was ported (with cross-referencing comments, which is why
  it is still correct in both), while step recording, suspension and token
  accounting were not. #16's per-call trajectory records are the newest thing
  the SSE path does not have.
  The structural problem is that streaming is an *output* concern being
  solved by duplicating *control flow*.
- **Fix**: applied, as the refactor it needed rather than three patches
  (patching the symptoms separately would have produced a fourth copy of the
  logic to keep in sync). `ReActLoop` now emits a run-event stream both
  endpoints consume, and the ~170-line hand-written loop in `app/server.py`
  is gone. Six steps, the first four behaviour-preserving by construction —
  each landed with the whole existing suite passing and **no test changed**:

  1. `agent/trigger/events.py` — `RunEvent` plus the effect vocabulary.
  2. `StateGraph.iter_steps()` yields `(name, node)` and takes the resulting
     state back, so the *caller* owns node execution while the graph keeps
     owning routing. `compile()` is now that plus a two-line runner, so the
     two cannot disagree about edges.
  3. Nodes became generators. This is the step that makes the whole thing
     work and the one the original plan hand-waved ("share the node
     implementations"): `_act_node` contained a blocking
     `dispatcher.dispatch(...)`, an async driver needs `await` there, and a
     `def` cannot `await`. So a node no longer performs its own I/O — it
     `yield`s `CallModel` / `CallTool` / `ManageContext` and is handed the
     result back. One node body, drivable by either driver. Exceptions are
     `throw()`n back in, so each node's own `try/except` around its I/O
     reads exactly as it did when the call was inline — which is where
     `TransientLLMError` still becomes a graceful stop and `SuspendRun` still
     becomes a checkpoint.
     `ManageContext` covers the call that is easiest to forget:
     `ShortTermMemory.manage()` summarizes the overflow window with a *model
     call*, so leaving it in the node body would have left the async driver
     with a blocking provider call it could not see.
  4. `_drive` / `_adrive` (about fifteen lines each) plus `_perform` /
     `_aperform`. The async side runs tool dispatch, context management,
     context-provider preparation and memory writes in `asyncio.to_thread`.
  5. `/api/stream` became a translator: `RunEvent` → `_sse(...)`.
  6. Deleted the duplicate, including the cross-referencing comments that
     existed only to warn that a copy existed. Their disappearance is the
     real deliverable.

  Measured after: the 300ms tool now delays a 30ms coroutine to **35ms**
  (was 342ms); a suspending run emits a `suspended` event carrying a
  `run_id` that `POST /api/runs/{run_id}/resume` accepts, with the
  checkpoint saved before the event is sent; `steps` and `tokens` come from
  the one `ExecutionContext` and a 4000-character prompt is counted rather
  than reported as 5 tokens.

  Two things the streaming path *gained* by no longer being a copy, neither
  of which was in scope: loop detection (a model repeating one identical
  call used to run to the full step budget there), and the same
  `stop_reason` vocabulary as `POST /api/run` (it used to invent a bare
  `"max_steps"` of its own).

  `Usage.estimated` was added in the same change: a streamed reply carries no
  usage block from any provider here, so both halves are estimated —
  reporting an estimate is fine, reporting it as measured is not.

  `app/server.py` is ~250 lines shorter net.
- **Regression test**: `tests/test_unified_execution.py` — one per symptom
  (`test_a_slow_tool_does_not_stall_the_event_loop`,
  `test_a_streamed_run_that_suspends_says_so_and_saves_a_checkpoint`,
  `test_the_streaming_path_reports_the_steps_it_actually_took`), plus the
  properties that had to survive: `test_run_is_exactly_a_drain_of_iter_run`,
  `test_the_sync_and_async_drivers_produce_the_same_run`, and
  `test_events_pair_up_one_per_tool_call`. Plus
  `test_the_streaming_path_now_detects_a_loop_too` in
  `tests/test_server_stream_incomplete_run.py` for the capability it gained.
- **Status**: Fixed. Design and rationale kept in
  `docs/unified-execution-core.md`.

---

## 23. The graph's transition cap fired before the step budget it was protecting

- **Found**: 2026-09-05, while unifying the execution core (#22) — a
  streaming run that should have reported `max_steps` reported
  `max_transitions (200)` instead. Pre-existing on **both** paths, not
  introduced by that work: `run()` has always driven the graph through
  `compile()`, whose cap is the same 200.
- **Symptom**: a long run stopped with
  `stop_reason = "max_transitions (200)"` — a number about the graph engine,
  not about the run. Nothing downstream could act on it: it is not a budget
  reason, not a loop, not a failure anyone configured.
- **Root cause**: `StateGraph.compile()`/`iter_steps()` take
  `max_transitions=200`, a safety net against a graph that cannot terminate.
  But a ReAct *step* costs two transitions (think, act) and three when a
  tool failure forces a reflection turn, so 200 transitions is roughly 66
  steps — and the shipped `leader.max_steps` is **100**. The engine's
  safety net was therefore tighter than the budget it was supposed to be
  protecting, and tripped first on any run that used its full allowance.
  Nothing caught it because no test ran a long enough loop: `MockLLM`
  answers within a couple of steps.
- **Fix**: `ReActLoop._max_transitions()` derives the cap from the budget —
  `max(200, max_steps * 4 + 20)` — and both `iter_run` and `aiter_run` pass
  it. Computed per run rather than at construction, because `max_steps` is
  writable (see BUGS.md #14). `ExecutionContext.over_budget()` remains the
  authoritative guard; the graph cap is back to being what it was meant to
  be, a net for a graph that cannot terminate at all.
- **Regression test**:
  `test_the_graph_transition_cap_never_fires_before_the_step_budget`
  (`tests/test_unified_execution.py`) — a model that never answers, with
  `max_steps=100`, must stop with `budget: max_steps (100) reached` and
  exactly 100 steps.
- **Status**: Fixed.

---

## 24. Six tests were spending real money on every local run

- **Found**: 2026-09-05, immediately after running a manual SSE sanity check
  that returned a real model's answer and a real `call_00_...` tool-call id.
- **Symptom**: `app/server.py` calls `load_dotenv()` at import, and
  `_auto_llm` selects DeepSeek the moment `DEEPSEEK_API_KEY` is present. So
  on any machine with a populated `.env` — the developer's — any test
  reaching a server endpoint without patching the model factory issued live,
  billed API calls. It passed either way, so nothing pointed at it. The
  external review's own run reported "340 passed, 5 skipped … no real model
  was called" precisely because it had disabled `.env` loading; the same
  suite here silently did the opposite.
- **Root cause**: the guard was per-test discipline (`monkeypatch.setattr(
  server, "_build_llm", ...)`) rather than a default. Discipline holds until
  someone adds a test that does not know it needs it — which is exactly what
  happened while writing #22's regression tests.
- **Fix**: `tests/conftest.py` — an autouse fixture pointing the server's
  model factories at `MockLLM` for every test. A test wanting a specific
  fake still overrides it, since its own `monkeypatch` runs later and wins.
  `_build_fast_llm` is patched to *delegate* to `_build_llm` rather than
  returning its own `MockLLM`, mirroring the real `models.fast:
  provider: auto` fallback — otherwise a test patching only `_build_llm`
  would silently lose the fast tier, the exact trap that function's docstring
  already warns about. (One test, `test_conversation_title_...`, caught this
  during the fix.)
  Deliberately narrow: the API keys are **not** unset. `test_tool_scaling*`
  and `test_hierarchical_agent` are *intentionally* live — they measure a
  real model's tool-selection accuracy, which no fake can stand in for — and
  gate themselves on a key being present. Those are a deliberate choice to
  spend money; this guards the accidental case.
- **Regression test**: the fixture is the guard, and it is autouse, so it
  applies to every test written from now on without anyone remembering it.
- **Status**: Fixed. Worth noting separately: those intentional live tests
  still run by default on any machine with a key configured, so a plain
  `pytest` there costs money. Making them opt-in (a marker, or
  `RUN_LIVE_LLM_TESTS=1`) is a reasonable follow-up but is a change to a
  documented design decision, not a defect.
