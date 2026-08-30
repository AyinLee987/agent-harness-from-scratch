# AGENTS.md — how this codebase is written

This file exists so that any agent (human or LLM) touching this repo produces
code that reads as if one person wrote it. It documents *observed* convention,
not aspiration — every rule below is backed by a real example already in the
tree. If you add something that doesn't fit a pattern here, prefer bending it
to fit over inventing a new one; if none fits, extend this file in the same
commit.

## 1. Why the codebase looks the way it does

Three convictions run through almost every module here. Know them before you
write anything, because most of the concrete rules below are just these three
ideas applied to a specific situation.

1. **State lives in exactly one place, explicitly.** `ExecutionContext` "is
   deliberately the *only* object that owns mutable run state"
   (`agent/state/context.py`). When you're tempted to stash a flag on `self`
   somewhere to remember something across calls, ask whether it belongs on
   the context object instead.
2. **Extensibility is a `Protocol` plus one in-memory reference
   implementation.** `SessionMemoryStore`/`InMemorySessionStore`,
   `MemoryRepository`/`InMemoryMemoryRepository`, `ContextProvider`,
   `MemoryPolicy` — every pluggable seam in this repo is a small structural
   `Protocol` (duck-typed, no inheritance required) with a zero-dependency
   in-memory implementation for tests and zero-setup usage, and a heavier
   backend (usually SQLite) for production. New extensibility points should
   follow the same shape, not a class hierarchy.
3. **Fail loud on setup, fail soft on side effects.** A missing API key
   raises immediately at construction (`BailianLLM.__init__`). A best-effort
   side effect — writing to long-term memory after a run completes — is
   wrapped in `try/except` so it can never take down the actual answer
   (`ReActLoop._record_memory_event`, with the comment explaining exactly
   why: *"Durable memory is best-effort here... answer delivery [should not]
   depend on an embedding service or vector index"*). When you add a step
   that isn't the actual point of the call, ask which category it's in and
   handle failure accordingly — don't let a logging call or a cache write
   turn into an unhandled exception on the user's request path.

## 2. Module and docstring voice

Every module's top docstring explains **why it exists**, not just what's in
it — read `compression.py`'s opening paragraph or `session.py`'s one-liner
(*"Conversation-level message storage separate from a single run
context."*) for the tone: a sentence or two of rationale, then get out of the
way. Avoid a docstring that just restates the class/function name.

Inline comments are rare and reserved for a non-obvious trade-off or a
deliberately unhandled edge case — and when used, they say so explicitly
rather than leaving it implied. Example, `react_loop.py`'s `_detect_loop`:

```python
"""
A known blind spot: a loop that *varies* its arguments each time (e.g.
``page=1``, ``page=2``, ...) is not caught here; that needs an
observation-novelty signal, not a call-identity one.
"""
```

If you make a deliberate simplification, say so in the same voice — a future
agent reading the code should be able to tell "this is a known limitation"
from "this is a bug."

## 3. Typing

- `from __future__ import annotations` at the top of every module.
- Prefer `typing.Optional[X]` / `List[X]` / `Dict[K, V]` over PEP 604 `X | None`
  — this is the codebase's overwhelming majority style. A few newer
  provider-adapter classes (`BailianLLM`, `DeepSeekLLM`) use `X | None`; don't
  spread that further, and don't "fix" it in unrelated files.
- Every public function and method signature is fully typed, including
  return type. Dataclasses and `Protocol` bodies typically don't need inline
  comments beyond the docstring — the type hints are the documentation.
- Structural typing over inheritance: `ContextProvider`, `SessionMemoryStore`,
  `MemoryRepository`, `EmbeddingProvider` are all `Protocol` classes.
  Implementations don't need to `import` or subclass them; they just need
  the right method signatures (see `adapters/conversation_history.py` in the
  companion eval project for a caller-side example of relying on this).

## 4. Naming conventions

- **IDs are short and generated the same way everywhere**:
  `uuid.uuid4().hex[:8]` for run-scoped ids (`run_id`, step ids as
  `f"{run_id}-{index}"`), `uuid.uuid4().hex[:12]` for entity ids (message
  ids, memory record ids). Never a full 36-character UUID — these ids show
  up in logs and are meant to be read, not just compared.
- **Every id is `<entity>_id`**: `run_id`, `conversation_id`, `record_id`,
  `candidate_id`, `subject_id`. Don't shorten to `id` on anything that isn't
  the primary key of its own table/dataclass.
- **Booleans and predicates read as a question or a completed fact**:
  `over_budget()`, `wants_tool` (property), `success`, `pinned`. Not
  `is_over_budget_flag` or `budget_ok`.
- **Chat messages are always the OpenAI wire-format dict** —
  `{"role": ..., "content": ..., **extra}` — never a custom `Message` class.
  This is deliberate: it's the one shape every LLM provider, every
  `ContextProvider`, and every test fixture already speaks, and introducing
  a wrapper type would mean translating at every boundary instead of none.
- **Closed sets of states are enums, not strings** — `MemoryStatus`,
  `MemoryDecision`, `MemoryKind`, `RetentionPolicy`. Open-ended or
  wire-format values (message `role`, tool `name`) stay plain strings.
- **Private helpers are `_leading_underscore`**, module-level or on a class,
  and are usually one clear verb phrase: `_detect_loop`, `_record_memory_event`,
  `_near_budget`.

## 5. Class shapes — pick the right one

| Shape | When | Examples |
|---|---|---|
| `@dataclass` | A value/result object, or mutable state owned by one component | `AgentResult`, `Step`, `Usage`, `LLMResponse`, `MemoryCandidate`, `SummarySnapshot` |
| `Protocol` | A pluggable seam with more than one real implementation | `ContextProvider`, `SessionMemoryStore`, `MemoryRepository`, `EmbeddingProvider` |
| Plain class, constructor-injected dependencies | A component with behavior, built once and reused | `ReActLoop`, `MemoryManager`, `ContextCompressor`, `ToolDispatcher` |
| Small stateless class implementing a `Protocol` structurally | An adapter/provider for one specific integration | `RAGContextProvider`, `SessionContextProvider` |

Constructor-injection is the rule for anything with a dependency: an LLM, a
tool registry, a compressor, an output guard, a memory manager are all passed
into `__init__`, all with sensible defaults where a zero-setup path makes
sense (`MockLLM`, `InMemorySessionStore`, `NoopMemoryExtractor`). Avoid
module-level singletons except at the application boundary — `app/server.py`'s
handful of `REGISTRY` / `RAG_PIPELINE`-style globals are the accepted
exception because a FastAPI app needs *some* place to hold process lifetime
state; application logic itself should not reach for a global.

## 6. Errors

Two-tier taxonomy, defined once in `errors.py` and reused everywhere a tool
can fail: `RecoverableToolError` (the model can fix its next action and
should see the failure as a tool observation) vs. `FatalToolError` (the run
must stop). Don't invent a third tier or a per-tool exception type — classify
into one of these two at the call site that can fail, the way `calculator`'s
`ZeroDivisionError` is caught and re-raised as `RecoverableToolError` in the
companion eval project's tool set.

Unclassified exceptions are treated as fatal by the dispatcher — so an
un-caught library exception inside a tool is a bug to fix (classify it),
not a gap to leave.

## 7. Observability

Structured, stdlib-only logging via `get_logger(__name__)` +
`log_event(logger, level, "namespace.action.phase", **fields)`. Event names
are dot-namespaced and read like a sentence fragment:
`agent.run.started`, `llm.call.completed`, `memory.write.skipped`,
`memory.write.pending_confirmation`. Follow `<subsystem>.<action>.<outcome>`
for anything new — don't invent a different separator or casing.

Correlation ids (`run_id`, `agent_name`) are bound for the duration of a
logical operation with the `bind_log_context(...)` context manager, not
passed as an explicit parameter to every log call down the stack.

`sanitize()` strips secrets (api keys, tokens, passwords) from anything
logged — call it on any new field that might carry user- or config-supplied
values, rather than trusting the field to be safe.

## 8. Guardrails are explicit, not assumed

Nothing here trusts the model to stop itself. `max_steps`, `max_tokens`,
loop detection, and tool-retry limits are all real, checked, testable
conditions on `ExecutionContext` — not a comment saying "the model should
stop when it's done." If you add a new loop or a new resource the agent can
consume unboundedly, add an explicit, testable limit for it in the same
change, following the existing `over_budget()` / `budget_reason()` shape.

## 9. Tests

- One test file per module (`tests/test_<module>.py`), pytest, plain
  `assert` — no assertion-library dependency.
- Fakes over mocking-framework mocks: a small subclass of `MockLLM`, or a
  minimal object satisfying a `Protocol` structurally, is preferred over
  `unittest.mock.Mock()` with configured return values (see
  `_DelegatingLeaderLLM` in `tests/test_server_delegation.py`).
- Server endpoints are tested by calling the FastAPI route function
  directly with `asyncio.run(...)` and `monkeypatch.setattr(server, ...)` to
  swap in a fake LLM — not by spinning up a `TestClient` for behavior that
  doesn't need real HTTP.
- A new feature earns both a new rubric/edge-case test *and*, where
  applicable, a regression test named after the bug it fixes — see
  `test_render_pads_none_valued_cells_so_columns_stay_aligned` in the
  companion eval project for the naming pattern: describe what the test
  guarantees, not what it calls.

## 10. Adding a new pluggable seam — checklist

When you add something a caller should be able to swap out (a new storage
backend, a new provider, a new policy):

1. Define a minimal `Protocol` with only the methods callers actually need.
2. Ship one dependency-free in-memory implementation of it.
3. If durability matters, add a SQLite implementation next to the in-memory
   one, following `SQLiteMemoryRepository`'s pattern: a version-tolerant JSON
   payload column plus indexed lookup columns, `check_same_thread=False`,
   an explicit `close()`, and a `threading.RLock()` around every statement.
4. Re-export the new names from the subpackage's `__init__.py` and from the
   top-level `agent/__init__.py`, keeping both the import block and
   `__all__` alphabetically sorted within their existing groups.
5. Wire it into `app/server.py` behind an environment variable following the
   `<THING>_DB_PATH` convention (see `RAG_DB_PATH`), defaulting to a real
   SQLite file under `data/`, not to in-memory — production should default
   to durable, tests should default to `InMemory*` or `:memory:`.
