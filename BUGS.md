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
