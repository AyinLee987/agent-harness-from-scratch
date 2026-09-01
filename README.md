# agent-harness-from-scratch

A minimal, production-shaped **ReAct agent framework** built from first principles in pure Python — no LangChain, no LlamaIndex. The goal is to show the *internals* of an agent runtime: the think–act loop, an execution context, a typed tool abstraction, memory, guardrails, and an evaluation harness.

[![CI](https://github.com/AyinLee987/agent-harness-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/AyinLee987/agent-harness-from-scratch/actions/workflows/ci.yml)

![ReAct Agent Playground](docs/app-trajectory.png)

> The interactive **ReAct Agent Playground** — watch the agent think, call a tool,
> observe the result, and answer, with live run stats. Source in [`web/`](web/).

## Live demo

An interactive **ReAct Agent Playground** that visualizes the think → act →
observe loop in the browser: <!-- TODO: paste your published demo URL here -->
**_(coming soon)_**. Source lives in [`web/`](web/) — run it locally with
`cd web && npm install && npm run dev`.

![Playground — initial state](docs/app-home.png)

## See it in action

The agent reasoning and calling tools (`python examples/basic_tools.py`):

![Agent reasoning and calling tools](docs/agent-run.png)

The evaluation harness scoring the agent over the sample tasks
(`python examples/run_eval.py`) — runs on the zero-dependency mock LLM:

![Eval scorecard](docs/eval-scorecard.png)

Query-aware context compression shrinking a document before the LLM call
(`python examples/context_compression.py`):

![Context compression](docs/context-compression.png)

## Why this exists

Most "agent" projects wire together a framework and call it a day. This one implements the pieces that actually matter in production:

- **ExecutionContext** — owns state, step history, and a token/step budget
- **Tool abstraction** — a `@tool` decorator that auto-generates JSON schemas from function signatures
- **Memory** — short-term context management (window + summarization), policy-controlled long-term vector recall, and multi-turn conversation continuity (see [Conversation continuity](#conversation-continuity))
- **Context compression** — query-aware compression that shrinks the input before the LLM call, inspired by recent long-context research
- **Guardrails** — max-step limits, finish/confidence checks, and safe handling of malformed tool calls
- **Tool-output safety** — an injection guard that screens untrusted tool output for indirect prompt injection
- **Multi-agent orchestration** — a Leader can spawn/await Worker subagents as ordinary tools, with isolated state and hard resource limits (see [Leader and Subagents](#leader-and-subagents))
- **Governed hybrid RAG** — an optional BM25 + dense + RRF retrieval layer with citations and staged, versioned ingestion (see [Governed hybrid RAG](#governed-hybrid-rag))
- **MCP tool integration** — remote MCP servers register as ordinary tools, no special-casing in the agent core
- **Observability** — a run id and per-step correlation ids thread through `ExecutionContext`
- **Evaluation harness** — runs the agent over a task set, logs full trajectories, and scores with rule-based, **trajectory-level**, and optional LLM-as-judge

It runs **with zero setup and no API key** thanks to a deterministic `MockLLM`, and switches to a real model when `OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, or Bailian credentials are provided (see `agent/llm.py` for `OpenAILLM` / `DeepSeekLLM` / `BailianLLM`, and `examples/deepseek_demo.py` / `examples/bailian_sqlite_demo.py`).

## Architecture

The agent is a `think → act → observe` loop. A single `ExecutionContext` threads
through every iteration and owns all mutable state (messages, scratch state, the
step-by-step trajectory, and the token/step budget). Tools are typed and
self-describing; memory is layered into short-term and long-term; and an optional
context compressor shrinks the input before each LLM call.

![Architecture](docs/architecture.png)

Each loop iteration:

1. **`think()`** — one LLM call (after short-term memory management) returning
   either tool calls or a final answer.
2. **`act()`** — dispatch the requested tool(s), capture observations, and feed
   them back into the transcript. Malformed calls are retried once, then fail
   cleanly.
3. The budget guard on `ExecutionContext` stops the loop at the step or token
   limit, so a run can never spin forever.

## Quickstart

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY, or run with the mock LLM
python examples/basic_tools.py
python examples/run_eval.py
```

Everything above works **without an API key** (it uses `MockLLM`). To run against
a real model, set `USE_OPENAI=1` and `OPENAI_API_KEY` in your `.env`.

### A 30-second taste

```python
from agent import MockLLM, ReActAgent, ToolRegistry, tool

@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    return str(eval(expression))  # the repo ships a *safe* evaluator instead

agent = ReActAgent(llm=MockLLM(), tools=ToolRegistry([calculator]))
print(agent.run("What is 23 times 17?").answer)
# -> Based on the tool result, the answer is: 391
```

## MCP tools

MCP servers connect once, expose their tools dynamically, and register as
ordinary BaseTool instances. The agent core does not special-case MCP:
ToolRegistry exposes the remote JSON schemas and ToolDispatcher calls the
generated proxies like local tools.

This project uses MCP Python SDK v2 and requires Python 3.10+. For the official
[Fetch MCP server](https://github.com/modelcontextprotocol/servers/tree/main/src/fetch),
install uv so it can run that v1-based server in an isolated environment:

~~~bash
pip install -r requirements.txt
pip install uv
python examples/mcp_fetch.py
~~~

The example discovers fetch__fetch and lets ReActAgent invoke it. Namespacing
prevents tools from different servers from colliding.

To expose Fetch through the FastAPI/Playground server, opt in through the .env
file after reviewing the internal-network access warning:

~~~dotenv
ENABLE_FETCH_MCP=1
~~~

Restart uvicorn after changing this value. The /api/tools response will then
include fetch__fetch.

~~~python
import sys

from agent import MockLLM, ReActAgent, ToolOutputGuard, ToolRegistry
from agent.mcp import MCPManager, MCPServerConfig, uv_tool_command

command, args = uv_tool_command("mcp-server-fetch")
fetch = MCPServerConfig.stdio(
    name="fetch",
    command=command,
    args=args,
    env={"PYTHONIOENCODING": "utf-8"},  # recommended on Windows
)

with MCPManager([fetch]) as mcp:
    registry = ToolRegistry()
    registry.register_many(mcp.tools())
    agent = ReActAgent(
        llm=MockLLM(),
        tools=registry,
        output_guard=ToolOutputGuard(),
    )
    result = agent.run("Fetch https://example.com and summarize it.")
~~~

MCPServerConfig.http connects Streamable HTTP servers. Headers, timeouts,
working directories, explicit stdio environment variables, and output-size
limits are configurable.

The Fetch server can access local and internal addresses. Only connect trusted
MCP servers, restrict deployment network access where appropriate, and keep
ToolOutputGuard enabled because fetched pages are untrusted model input.

## Evaluation

Run `python examples/run_eval.py` to score the agent over the sample tasks
(`agent/eval/tasks.json`). Add `--judge` for the optional LLM-as-judge pass and
`--dump results.json` to write full trajectories. Example scorecard (MockLLM):

| Metric | Value |
|---|---|
| Tasks | 15 |
| Success rate (rule-based) | 100% |
| Success rate (LLM-as-judge) | 100% |
| Avg steps/task | 2.00 |
| Avg tokens/task | ~253 |

Scoring is layered:

- **Rule-based** — expected substrings present, the expected tool was actually
  invoked, and the agent finished cleanly rather than being force-stopped.
- **Trajectory score** — grades *how* the answer was reached (finished cleanly,
  used the expected tool, no tool errors), not just the final output. This
  catches "right answer via the wrong path" that output-only scoring misses.
- **LLM-as-judge** (optional) — for open-ended answers, with a judge/rule
  **agreement** metric reported as a cheap calibration proxy.

Numbers above are from the deterministic mock; with a real model they reflect
that model's quality.

### Tool-count scaling experiment

A common question when a project's tool registry grows is: *how many tools
before the model's tool-*selection* accuracy drops?* `examples/tool_scaling_kit.py`
ships a 50-tool catalog (five categories of ten near-duplicate tools each —
`math_*`, `text_*`, `date_*`, `convert_*`, `data_*` — the kind of overlap that
actually confuses selection, not just raw count) plus a matching
`examples/tool_scaling_tasks.json` (one task per tool).

`examples/tool_scaling_test.py` runs a fixed probe set of prompts unchanged
while padding the surrounding registry with pure distractor tools from 6 up
to the full 50, and reports selection accuracy at each size:

```bash
python examples/tool_scaling_test.py                 # DeepSeek by default
python examples/tool_scaling_test.py --provider bailian --sizes 6,15,25,35,50
```

Requires a real LLM (`DEEPSEEK_API_KEY` / `BAILIAN_API_KEY` / `OPENAI_API_KEY`)
— MockLLM's tool choice is keyword-heuristic, not a model decision, so it can't
exhibit this effect. `tests/test_tool_scaling.py` covers the kit itself
(50 unique, well-formed tools; golden-value checks on representative tools)
unconditionally, plus a live-LLM smoke test that's skipped without an API key.

Sample result (DeepSeek-chat, 6 fixed probe tasks): accuracy held at **100%
from 6 tools up to the full 50** for this kit/model — a useful data point
against the common "~20 tools" folklore threshold, which appears to bite
sooner mainly with smaller/weaker models, far larger tool counts (100s), or
tools with even closer name/argument overlap than this kit's.

Single-tool selection is the easy case, though. `examples/tool_scaling_multi_test.py`
raises the bar with **chained** tasks against the same 50-tool registry —
`examples/tool_scaling_multi_tasks.json` has 14 tasks each requiring several
tool calls in a specific order (some reusing the *same* tool twice, e.g.
"add 12 and 8, then add 15 to that"), where one wrong pick anywhere in the
chain fails the whole trajectory:

```bash
python examples/tool_scaling_multi_test.py
```

*(Historical result, when this file's chains were 2 steps long: 14/14 exact
tool-sequence matches (100%) at 50 tools. The tasks were later extended to
5+ steps each — see "All chains extended to 5+ steps" below for the current
result, which looks very different.)*

Neither raised tool count nor chained calls surfaced degradation, so the
last variable from the original framing — *description length*, independent
of tool count — gets isolated next. `examples/tool_scaling_verbose_kit.py`
wraps the same 50 tools (identical names/params/behavior) with descriptions
padded ~6.5x longer using deliberately repetitive boilerplate (the realistic
failure mode: a safety/usage paragraph pasted into every tool's docstring,
which drowns each tool's actually-distinguishing text in near-identical
filler). `examples/tool_scaling_verbose_test.py` reruns both the single-tool
probe set and the 14-task chain against this verbose registry:

```bash
python examples/tool_scaling_verbose_test.py
```

Sample result (DeepSeek-chat, 50 tools, 14,705 → 94,975 schema chars, 6.5x):
single-tool probe accuracy **100% (6/6)**, chain accuracy **100% (14/14 at
the time, on the since-extended-to-5-steps chain file)** — unchanged from
the concise baseline. Across all three axes tested at 50 tools (tool count,
chained calls, ~6.5x description bloat), this model/kit combination showed
no measurable accuracy drop.

### Pushing tool count to 100

The kit was doubled to **100 tools** (five more categories — `stat_*`,
`format_*`, `calendar_*`, `measure_*`, `encode_*` — each extending an
existing category's spirit, e.g. `stat_median`/`stat_stdev` alongside
`math_average`) to see if 50 simply wasn't enough tools to bite. Rerunning
both experiments at the new size:

```bash
python examples/tool_scaling_test.py --sizes 6,15,25,35,50,75,100
python examples/tool_scaling_multi_test.py
```

Single-tool probe accuracy: still **100%** at every size up to 100. The
multi-tool chain (2 steps at the time), though, cracked slightly: **13/14
(93%) exact tool-sequence matches** — one task (add 5 days, then add 20
more) had the model insert an unrequested *third* `date_add_days` call as a
self-check before answering. The final answer was still correct (14/14,
100%), so this wasn't a wrong-tool-selection failure yet — it's the first
hint that longer trajectories give the model room to deviate.

### Chain length, not just tool count

Every chained task at that point was only 2 steps. `examples/tool_scaling_long_chain_tasks.json`
first added 7 tasks at 3-5-step chains to ask whether *chain length itself*
— independent of registry size — is a lever, and got a first real (if
narrow) signal: one 3-step task had the model answer a weekend question
correctly from its own reasoning ("Yes, it's a weekend") without calling the
required `calendar_is_weekend` tool — right answer, skipped step.

### All chains extended to 5+ steps

Both chain task files were then rewritten so *every* task is a 5+ step
chain — no more short chains anywhere in this kit's test suite:

```bash
python examples/tool_scaling_multi_test.py       # 14 tasks, all 5 steps
python examples/tool_scaling_long_chain_test.py  # 7 tasks, all 5 steps
```

This is where the effect finally shows up cleanly. **Both files landed at
the same number independently: 57% exact-sequence match** (8/14 and 4/7),
down from 93-100% at shorter chains — while **final-answer accuracy stayed
100% in both runs.** The mechanism is consistent across every miss, in both
files: the model quietly **skips a step it judges redundant**, most often a
`math_round` call whose input is already at (or past) the target precision
— e.g. asked to round `70` to 0 decimals, it just reports `70` without
calling the tool — and in one case (the weekend task, again) it drops
`calendar_is_weekend` entirely, inferring the answer instead of calling the
required tool. Every skip left the final numeric/factual answer unchanged,
because a skipped no-op round doesn't change the value — which is exactly
why final-answer accuracy can stay perfect while trajectory-exactness
collapses. This is the clearest, most reproducible signal in the whole kit
so far: not "picks the wrong tool," but "silently shortens a long,
instruction-following-heavy trajectory once it's confident it already has
the answer" — and it appears to kick in specifically once chains reach
~5 steps, not at 2-4.

### Hierarchical routing: main agent + specialist subagents

The standard mitigation for "too many tools" in production is namespacing:
give the main agent only a handful of base tools plus one short-description
"delegate" tool per specialist, and let each specialist run as its own
`ReActAgent` with its own small, fully-described tool set. `examples/
hierarchical_agent_kit.py` implements this — main agent sees `fetch`,
`get_date`, and five one-line `delegate_*` tools (`calculate_agent`,
`text_agent`, `datetime_agent`, `convert_agent`, `data_agent`, 20 kit tools
each); its own schema drops from 30,630 to 2,728 chars (11.2x smaller) than
the flat 100-tool registry. A specialist that can't complete its sub-task
with what it has must reply `TASK_FAILED: <reason>` instead of guessing.

`examples/hierarchical_agent_test.py` reran the *exact same* 21 five-step
tasks that scored 57% exact-sequence match flat, through this hierarchy
instead — flattening every specialist's own tool calls, in delegation
order, back into one sequence for an apples-to-apples comparison:

```bash
python examples/hierarchical_agent_test.py
```

**Exact-sequence accuracy rose to 67% (14/21), up from the flat 57%** — but
**final-answer accuracy dipped slightly to 95% (20/21)**, and the win came
with real overhead: 85 delegate calls across 21 tasks (avg 4.05/task,
vs. the flat trajectory's own 5 tool calls), so roughly 2x the LLM round
trips for a 10-point gain. Digging into which tasks flipped:

- **5 flat misses got fixed** (`multiply_twice`, `add_days_then_days_between`,
  `sum_div_round`, `add_days_weekday_weekend`, `sum_div_round_temp_round`) —
  each specialist call only has to carry a 1-2-step sub-chain, and a
  shorter, narrower sub-task seems to make the "skip the redundant round"
  tendency less likely to fire.
- **4 misses persisted unchanged** (`celsius_roundtrip`, `add_twice`,
  `replace_then_wordcount`, `add_days_twice`) — same mechanism as the flat
  run: a specialist still drops a `math_round`/`calendar_is_weekend` call
  it judges redundant, just now inside its own delegated sub-chain instead
  of the top-level one.
- **3 new misses appeared** (`add_days_then_weekday`, `average_then_round`,
  `snake_reverse_vowels`) — mostly the same redundant-round skip recurring
  (a stochastic tendency, not new), plus one genuinely new failure mode:
  `add_days_then_weekday`'s main agent delegated "count the characters"
  to `text_agent` (`text_char_count`, got 8), then **delegated the same
  already-answered sub-task again** to `calculate_agent` (`stat_count`,
  got 8 again) — group boundaries created an ambiguity (which specialist
  "owns" character counting?) that the flat registry never had to resolve,
  costing an extra, redundant call.
- The final-answer dip is the same known **grading-artifact**, not a new
  defect: `lower_snake_reverse_count_leap`'s main agent correctly says
  "18 is not a leap year" but the substring check wants the literal token
  `false` — identical to the flat run's known limitation, see above.

Net read: for *this* kit's regime (100 tools, 5-step chains), hierarchical
routing is a real, measured win on trajectory-exactness (worth it if you
actually have hundreds of real tools and want to keep every call's schema
small) but not a free one — it trades some latency/cost and introduces its
own new failure surface (redundant cross-specialist delegation) in exchange,
and doesn't touch final-answer accuracy, which was already ~100% either way.
One run of 21 tasks is not a large sample — treat 57% vs 67% as a real,
reproduced-once signal, not a tight confidence interval.

### Trying the scaling kit live (web UI / API)

The experiments above run standalone (`python examples/tool_scaling_*.py`) —
by default the running agent server (`app/server.py`, what the web UI and
`/api/*` talk to) still only has its normal handful of tools. To register
the kit onto the *live* agent instead, set in `.env`:

```bash
ENABLE_TOOL_SCALING_KIT=1     # adds the kit's tools to the live registry
TOOL_SCALING_KIT_VERBOSE=0    # 1 for the ~6.5x-longer descriptions
TOOL_SCALING_KIT_SIZE=100     # how many of the 100 kit tools to register
```

Off by default — leave unset for normal use. Check what actually landed with
`GET /api/tools`.

## Tool-output safety

Tool results are untrusted input — a web page or API response can carry text like
*"ignore your previous instructions and …"*. Left unchecked, that flows into the
model's context as **indirect prompt injection**. `ToolOutputGuard` gives the
agent a single choke point: it screens every observation for injection directives
and neutralizes them before they reach the model.

```python
from agent import ReActAgent, ToolOutputGuard

agent = ReActAgent(llm=..., tools=..., output_guard=ToolOutputGuard())
```

See `examples/prompt_injection.py` for a before/after demo.

## Context compression

Long-context agents waste most of their tokens re-sending tool outputs and
documents the model has already seen. Recent research shows that aggressively
compressing the input *before* it reaches the model preserves task accuracy while
cutting compute and latency — *Latent Context Language Models* (Chari et al.,
2025) report up to **16× compression** by compressing the input sequence ahead of
the decoder, and *ACON* targets this for long-horizon LLM agents specifically.

`ContextCompressor` implements a lightweight, model-free approximation of that
idea: a **selective, query-aware extractive compressor**. It splits context into
units, scores each for relevance to the current query, and keeps only the
highest-value units up to a target ratio — no trained encoder, fully
deterministic, zero extra dependencies.

```python
from agent import ContextCompressor

compressor = ContextCompressor(target_ratio=4.0)
result = compressor.compress(long_document, query="Why were there shipping delays?")
print(result.summary())   # e.g. "229->58 tokens (3.9x, kept 3/12 units)"
```

Wire it into an agent and it transparently compresses large message bodies before
each `think()` call:

```python
agent = ReActAgent(llm=..., tools=..., compressor=ContextCompressor(target_ratio=4.0))
```

See `examples/context_compression.py` for a runnable demo.

## Design notes

**Why ReAct.** Interleaving reasoning and acting keeps the agent grounded: every
action produces an observation that conditions the next thought, which is far
more robust than asking a model to plan an entire tool sequence up front. It also
makes the loop trivially inspectable — each step is a `(thought, action,
observation)` triple you can log and replay.

**Why an explicit context object.** Mutable run state is the thing that bites you
in production: it leaks across requests, makes runs impossible to reproduce, and
hides the token/step budget. Putting *all* of it in one `ExecutionContext` means
there is exactly one place to serialize for logging, one place to enforce the
budget, and one place to reset between tasks (the eval harness builds a fresh
agent per task for exactly this reason).

**How the budget guard prevents runaway loops.** The loop checks
`ctx.over_budget()` before every step, and both the step count and cumulative
token total are hard caps. A model that keeps calling tools, or keeps emitting
malformed calls, hits the ceiling and the run ends with a graceful "stopped"
answer plus the last useful observation — never an infinite spend.

**How memory is layered.** `ExecutionContext` owns one run, short-term memory
manages its prompt window, and `SessionMemoryStore` is the boundary for a future
conversation backend. Durable memory is policy-controlled: completed model
answers are never persisted implicitly. `MemoryRepository` is the source of
truth while the vector index is derived and rebuildable. Authoritative domain
knowledge belongs in a separate RAG corpus, not user memory.

**Why compress context.** Token cost and latency in agent loops are dominated by
re-sending history and tool outputs. Compressing the input before the model call
— rather than only summarizing across turns — keeps the budget flat as
trajectories grow. The compressor lives behind a small `compress()` interface, so
the model-free extractive implementation here can be swapped for a trained
encoder (the approach the LCLM/ACON papers take) without touching the agent.

**Treat tool output as untrusted.** Anything a tool returns may be
attacker-controlled, so the agent funnels every observation through one guard
before it reaches the model. Centralizing it (rather than sprinkling checks
through tools) means there's a single place to harden and test the boundary.

**Score trajectories, not just outputs.** A correct final answer reached via the
wrong tool, an error it recovered from, or a lucky guess all look identical to
output-only scoring. Grading the trajectory — did it finish cleanly, use the
right tool, avoid errors — is what makes the eval harness catch regressions
instead of vibe-checking behavior.

**What I'd change to scale this.** Make tool dispatch async so independent calls
run concurrently; past demo scale, default to `ChromaVectorStore` (HNSW
indexed, not brute-force like the NumPy/SQLite backends) or a hosted
Qdrant/pgvector — all plug into `LongTermMemory` and `DenseRetriever`
identically via `BaseVectorStore`, this was already made pluggable rather
than something still to build; swap the extractive compressor for a
trained context encoder; add streaming and partial-result handling; and introduce
a planner/executor split for multi-step tasks so a higher-level agent decomposes
work and delegates to focused sub-agents.

## Architecture: Trigger Layer + State Layer

The agent is split into two clear layers:

- **Trigger Layer** (`agent/trigger/`) — decides **when** and **how** the agent acts
- **State Layer** (`agent/state/`) — manages **what** the agent knows

A **Gateway** sits at the entry point providing rate limiting, concurrency
control, and request queuing for production deployments.

## Project layout

```
agent/
  memory/            ← Policy-controlled durable memory
    models.py        candidates, records, versions, retention states
    manager.py       read/write/lifecycle coordinator
    embeddings.py    chat-independent EmbeddingProvider
    repository.py    in-memory + SQLite source-of-truth repositories
    index.py         replaceable derived vector index
    policy.py        fail-closed extraction and persistence policy
    session.py       🆕 SessionMemoryStore: in-memory + SQLite conversation storage
    context.py       🆕 SessionContextProvider: replays a session into a stateless run
  trigger/           ← Trigger Layer (when / how)
    gateway.py       🆕 Unified entry: rate limiting + concurrency + queuing
    graph.py         Generic StateGraph engine (pattern-agnostic)
    react_loop.py    🆕 ReAct think→act→observe cycle (extracted from agent.py)
    dispatch.py      🆕 Tool execution with retry logic
  state/             ← State Layer (what)
    context.py       ExecutionContext: messages, steps, budget, run_id
    memory.py        ShortTermMemory + LongTermMemory (vector recall)
    store.py         BaseVectorStore → NumPy / SQLite / (FAISS / Qdrant future)
    chroma_store.py  ChromaVectorStore — optional dep, HNSW-indexed
  multi_agent/       ← Leader/Worker delegation
    orchestrator.py  MultiAgentOrchestrator: spawn/await/cancel Worker tasks
    registry.py      AgentRegistry: named Worker factories (fresh agent per task)
    tools.py         spawn_subagent / get_subagent_status / wait_subagents / cancel_subagent
    models.py        RunBudget, SubagentTask/Result, TaskStatus
  rag/               ← Governed hybrid RAG (see "Governed hybrid RAG" below)
    chunking.py      structure-first parent/child chunker
    ingestion.py      staged ingestion, dedup, versioned publication
    retrieval.py     BM25Retriever + DenseRetriever
    pipeline.py      RAGPipeline: fusion + rerank + parent-context hydration
    rerank.py        HeuristicReranker / CallableReranker
    query.py         MedicalQueryPlanner
    context.py       RAGContextProvider + create_rag_search_tool
    repository.py    in-memory + SQLite document/chunk storage
    models.py        Document, Chunk, Citation, Evidence(Bundle) types
  local_tools/       ← opt-in workspace file + CLI tools (see "Local workspace and CLI tools")
    tools.py         ReadFileTool, WriteFileTool, ListFilesTool, RunCommandTool
  context.py         ContextProvider protocol: pluggable pre-model-call hooks
  errors.py          ToolCallError taxonomy: RecoverableToolError / FatalToolError
  tools.py           BaseTool + @tool decorator (auto JSON schema) + ToolRegistry
  mcp/               persistent MCP clients + dynamic BaseTool proxies
  llm.py             LLM clients: MockLLM, OpenAI, DeepSeek, Bailian
  compression.py     ContextCompressor: query-aware context compression
  safety.py          ToolOutputGuard: indirect prompt-injection defense
  observability.py   structured logging, correlation-id context, secret redaction
  agent.py           Thin facade: wires trigger + state + shared infrastructure
  eval/
    harness.py       runs tasks; rule-based + trajectory + LLM-as-judge scoring
    tasks.json       sample eval tasks with expected outcomes
app/                 ← entry points, run from the repo root
  server.py          FastAPI app: /api/run, /api/stream, /api/tools, /api/rag/documents
  rag_ingest.py      CLI for local/batch RAG corpus ingestion (python -m app.rag_ingest)
examples/
  basic_tools.py         calculator + web-search-stub + datetime + memory_search tools
  context_compression.py query-aware context compression demo
  prompt_injection.py    tool-output injection guard before/after demo
  mcp_fetch.py            official Fetch MCP registered as fetch__fetch
  memory_demo.py         long-term vector memory recall demo
  medical_rag.py         governed hybrid RAG pipeline demo
  deepseek_demo.py       running the agent against DeepSeek instead of MockLLM
  bailian_sqlite_demo.py running against Bailian with SQLite-backed memory/RAG
  run_eval.py            runs the eval harness and prints a scorecard
tests/
  test_agent.py                    unit + integration tests (run entirely on MockLLM)
  test_graph.py                    StateGraph engine tests
  test_compression.py              tests for the context compressor
  test_safety.py                   tests for the prompt-injection guard
  test_observability.py            structured logging + correlation id tests
  test_mcp.py                      MCP discovery, proxy, lifecycle, and result tests
  test_memory_manager.py           MemoryManager lifecycle + session store tests
  test_multi_agent.py              Leader/Worker orchestration tests
  test_rag.py                      chunking, retrieval, RRF fusion, rerank tests
  test_rag_ingestion_endpoint.py   the RAG document-upload API endpoint
  test_local_tools.py              workspace file tools + allowlisted CLI tool
  test_server_delegation.py        delegation as an ordinary Leader tool, via /api/*
  test_server_conversation.py      🆕 conversation_id continuity on /api/run
  test_session_context.py          🆕 SessionContextProvider replay + summarization
web/
  ReAct Agent Playground — interactive browser demo (Vite + React + TS)
```

## Roadmap

- [x] Trajectory-level eval scoring + judge/rule calibration
- [x] Indirect prompt-injection defense for tool output
- [x] Correlation ids through the execution context
- [x] Trigger / State layer architecture separation
- [x] Gateway with rate limiting + concurrency control
- [x] Leader/Worker multi-agent delegation (synchronous dispatch; see below)
- [x] MCP tool integration (stdio + Streamable HTTP)
- [x] Policy-controlled durable memory (`MemoryManager`), replacing implicit auto-save
- [x] Governed hybrid RAG (BM25 + dense + RRF, citations, versioned ingestion)
- [x] Opt-in local workspace file tools + allowlisted CLI tool
- [x] Multi-turn conversation continuity (`conversation_id` on both `/api/run` and `/api/stream`)
- [x] Conversation history/title endpoints + a playground sidebar for browsing and switching between conversations
- [ ] Async multi-agent orchestration (Worker dispatch currently runs via `asyncio.to_thread`, not a native async tool loop)
- [ ] Persistent vector memory (FAISS/Qdrant)
- [ ] Trained context encoder (LCLM/ACON-style) behind the compressor interface
- [ ] Per-task regression suite generated from production failures

## Tool error classification

Tool failures are classified where the failing operation is implemented:

```python
from agent import FatalToolError, RecoverableToolError, tool

@tool
def lookup(key: str) -> str:
    if not key:
        raise RecoverableToolError("key is required")  # returned to the model
    if database_is_corrupt():
        raise FatalToolError("lookup database is corrupt")  # aborts this run
    return perform_lookup(key)

@tool(error_policy="recoverable")
def remote_lookup(key: str) -> str:
    return call_remote_service(key)  # every unexpected operation error is recoverable
```

Unclassified exceptions are fatal by default. This fail-closed behavior makes
tool authors decide explicitly whether the model can repair a failure by
changing its next action.

## Leader and Subagents

`MultiAgentOrchestrator` exposes Worker lifecycle operations as ordinary tools
to a Leader. Each registry factory must create a fresh `ReActAgent`, which keeps
Worker conversations and mutable memory isolated during parallel execution.

```python
from agent import (
    AgentRegistry, AgentSpec, MultiAgentOrchestrator, ReActAgent,
    RunBudget, ToolRegistry,
)

workers = AgentRegistry()
workers.register(
    AgentSpec("researcher", "Fetches and summarizes source material."),
    lambda: ReActAgent(llm=llm, tools=research_tools),
)

with MultiAgentOrchestrator(
    workers,
    RunBudget(max_subagents=6, max_parallel_tasks=3),
) as orchestrator:
    leader_tools = ToolRegistry(orchestrator.leader_tools())
    leader = ReActAgent(llm=leader_llm, tools=leader_tools)
    result = orchestrator.run_leader(leader, "Research and compare the options")
```

The Leader receives `spawn_subagent`, `get_subagent_status`,
`wait_subagents`, and `cancel_subagent`. A child fatal error terminates only
that child and is returned as structured data; orchestration invariant failures
remain fatal to the root run. Root completion cancels orphan tasks, and hard
limits cover task count, parallelism, duplicate dispatches, depth, and Worker
execution time.

The normal FastAPI `/api/run` and `/api/stream` entry points always run this
Leader. Delegation is not a separate mode: `spawn_subagent` and the other
lifecycle operations are simply part of the Leader's tool list. The model uses
them only when the task benefits from Worker specialization or parallelism.
The built-in server registers `researcher` and `analyst` Worker roles.

## Policy-controlled memory

Passing legacy `LongTermMemory` to an agent no longer auto-saves every task and
model answer. New code should use `MemoryManager`; its safe default extractor
stores nothing.

```python
from agent import (
    ExplicitRequestMemoryExtractor, LLMEmbeddingProvider,
    MemoryManager, MockLLM, ReActAgent, ToolRegistry,
)

llm = MockLLM()
memory = MemoryManager(
    embedding_provider=LLMEmbeddingProvider(llm, model_id="demo:hash-v1"),
    extractor=ExplicitRequestMemoryExtractor(),
)
tools = ToolRegistry([memory.as_search_tool(
    namespace="user-memory", subject_id="user-123",
)])
agent = ReActAgent(
    llm=llm,
    tools=tools,
    memory_manager=memory,
    memory_namespace="user-memory",
    memory_subject_id="user-123",
)
```

Normal explicit memories may persist; health and secret candidates enter
`REQUIRE_CONFIRMATION`. Model output and failed runs are skipped. Replacements
create a new version and mark the old record `SUPERSEDED`, retaining history
while removing it from recall.

Lifecycle states are `ACTIVE`, `SUPERSEDED`, `EXPIRED`, `TOMBSTONED`, and
`QUARANTINED`. Only `EPHEMERAL` and `TTL` records auto-expire. `PINNED`,
`UNTIL_REVIEW`, `UNTIL_SUPERSEDED`, and `EXPLICIT_DELETE_ONLY` records are not
removed by scheduled expiry. Deletion is two-phase: tombstone and de-index
immediately, then physically purge after an application-defined grace period.
Pinned deletion requires explicit authorization.

`EmbeddingProvider` is separate from the chat model and records its model id and
dimension on each memory. The old `LongTermMemory(llm)` constructor remains as a
compatibility adapter; production code should configure a dedicated embedding
provider and rebuild/version indexes when changing models.

## Conversation continuity

`ReActLoop.run()` builds a brand-new `ExecutionContext` on every call — by
itself it has no memory of a prior call, on purpose (see
`agent/trigger/react_loop.py`). Both `POST /api/run` and `GET /api/stream`
support continuing a conversation across separate calls by passing a
client-generated `conversation_id`:

```python
import requests

r1 = requests.post("http://localhost:8000/api/run", json={
    "task": "My name is Zhuoyang.",
    "conversation_id": "conv-123",   # any id you generate; reuse it to continue
}).json()

r2 = requests.post("http://localhost:8000/api/run", json={
    "task": "What's my name?",
    "conversation_id": "conv-123",
}).json()
print(r2["answer"])  # -> mentions Zhuoyang
```

`GET /api/stream?task=...&conversation_id=conv-123` works the same way — the
Agent Playground (`web/playground.html`, served at `GET /`) uses exactly this
to give the browser UI real multi-turn conversations: it keeps a sidebar of
known `conversation_id`s (tracked client-side in `localStorage`, since the
server intentionally has no "list all conversations" endpoint —
`SessionMemoryStore` is keyed by id, not enumerable), lets you switch between
them, and labels each entry with an LLM-generated **conversation title** —
see below.

Omit `conversation_id` and either endpoint behaves exactly as before — fully
stateless, nothing persisted. This is opt-in on purpose: the server never
starts keeping state you didn't ask it to keep.

Two read-only endpoints support a history/sidebar UI without adding any new
storage:

- `GET /api/conversations/{conversation_id}` — the stored turns for that
  conversation (empty list, not 404, for an id nothing has been sent under
  yet — a client-generated id the user hasn't used is not an error).
- `GET /api/conversations/{conversation_id}/title` — a short title
  summarizing the conversation, generated fresh from its stored messages on
  every call (nothing is persisted here — a second call can word it
  differently). The playground caches the first one it gets per conversation
  client-side rather than re-requesting it on every render.

Under the hood this is `SessionContextProvider` (`agent/memory/context.py`)
replaying a conversation's stored messages through the existing
`ContextProvider` hook, backed by `SessionMemoryStore` — `InMemorySessionStore`
by default (e.g. when a test calls a route function directly, bypassing app
startup), swapped for a durable `SQLiteSessionStore` at `data/sessions.sqlite`
(override with `SESSION_DB_PATH`) once the FastAPI app actually starts. Only
each turn's user message and final answer are persisted, not the
intermediate tool-call trajectory — a later turn replays this as plain
conversation, not a raw execution log (that's what `AgentResult.trajectory`
is for). Once a conversation passes 24 stored messages, older messages are
folded into a single persisted summary instead of being replayed (and
re-summarized by `ShortTermMemory`) in full on every subsequent turn.

## Structured logging

The FastAPI server configures structured logging automatically. Every request
gets an `X-Request-ID` response header, and the same id is attached to Leader,
Worker, LLM, tool, and MCP events. Multi-agent events additionally carry
`root_run_id`, `run_id`, `task_id`, and `agent_name` where applicable.

```dotenv
AGENT_LOG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
AGENT_LOG_FORMAT=json         # json or text
AGENT_LOG_FILE=logs/agent.log # optional rotating file; stderr is always enabled
AGENT_LOG_MAX_BYTES=10485760
AGENT_LOG_BACKUP_COUNT=5
```

Library users can initialize the same setup with `configure_logging()`. Stable
event names include `http.request.*`, `agent.run.*`, `llm.call.*`,
`tool.call.*`, `mcp.*`, and `subagent.*`. Logs record sizes, argument key names,
token counts, statuses, and timings—not prompt text, tool arguments, model
answers, or tool output. Known credential fields such as `api_key`,
`authorization`, `password`, and access tokens are redacted automatically.

## Governed hybrid RAG

The optional `agent.rag` package is a separate knowledge layer rather than a
form of user memory. It provides:

- structure-first Chinese medical parent/child chunks, including table-header
  carry-over and protected recommendation, evidence-grade, contraindication,
  and dose units;
- staged ingestion, checksum deduplication, atomic publication, version
  supersession, and SQLite persistence;
- BM25 plus dedicated dense embeddings, reciprocal-rank fusion, a replaceable
  reranker, metadata filters, and parent-context hydration — `DenseRetriever`
  delegates vector storage to a `BaseVectorStore` (default: in-memory
  `NumPyVectorStore`, same as `LongTermMemory`), so swapping in
  `SQLiteVectorStore` for persistence, or `ChromaVectorStore` for a real
  HNSW-indexed backend (optional `chromadb` dependency; a future
  Qdrant/pgvector backend would plug in the same way), is a constructor
  argument (`DenseRetriever(repo, embeddings, vector_store=...)`), not a
  rewrite — see `tests/test_vector_store.py` for the backend-agnostic
  contract test suite, parametrized over all three backends;
- traceable evidence IDs and source citations, plus insufficient, stale,
  conflicting, degraded, and retrieval-failed states;
- mandatory retrieval before the Leader's first model call and a recoverable
  `medical_evidence_search` tool for follow-up searches.

It is disabled by default. To enable it, copy UTF-8 Markdown or text documents
into `data/rag_sources`, then configure:

```dotenv
ENABLE_RAG=1
RAG_SOURCE_DIR=data/rag_sources
RAG_DB_PATH=data/rag.sqlite
RAG_EMBEDDING_MODEL=text-embedding-3-small
RAG_EMBEDDING_API_KEY=...
RAG_EMBEDDING_BASE_URL=https://api.openai.com/v1
```

The embedding endpoint is deliberately independent from the chat model. In a
medical deployment, replace `HeuristicReranker` with a validated cross-encoder,
set explicit evidence-age and multi-document policies through `RAGConfig`, and
ingest reviewed source metadata rather than relying on filenames alone.

Corpus writes have separate management entry points. For local or batch import:

```bash
python -m app.rag_ingest ./medical-guidelines \
  --publisher "Chinese Medical Association" \
  --document-type guideline --jurisdiction CN --version 2026
```

For an application upload, set a secret `RAG_ADMIN_TOKEN` and call:

```http
POST /api/rag/documents
X-RAG-Admin-Token: your-secret
Content-Type: application/json

{
  "logical_id": "guidelines/hypertension",
  "title": "Hypertension guideline",
  "content": "# Recommendation\n...",
  "publisher": "Reviewed publisher",
  "document_type": "guideline",
  "jurisdiction": "CN",
  "version": "2026"
}
```

The API accepts document text rather than a server-side file path, preventing
remote callers from reading arbitrary host files. It is disabled without the
admin token, enforces `RAG_MAX_DOCUMENT_BYTES`, serializes publication for
consistent versioning, and never publishes a document if chunking or indexing
fails.

## Local workspace and CLI tools

Local capabilities are ordinary Leader tools, so the existing `/api/run` and
`/api/stream` flows need no special endpoint. They are disabled by default:

```dotenv
ENABLE_LOCAL_FILE_TOOLS=1
AGENT_WORKSPACE_ROOT=C:\work\my-project

# Stronger capability: only enable for a trusted local user.
ENABLE_LOCAL_CLI=1
AGENT_CLI_ALLOWED_COMMANDS=git,rg,python,python.exe,pytest
AGENT_CLI_TIMEOUT_SECONDS=30
AGENT_CLI_MAX_OUTPUT_BYTES=262144
```

When enabled, the Leader receives:

- `read_file(path, start_line, end_line)` for bounded UTF-8 reads;
- `write_file(path, content, overwrite, create_parent_dirs, expected_sha256)`
  for atomic writes with optional optimistic concurrency checks;
- `list_files(path, pattern, max_results)` for workspace-scoped discovery;
- `run_command(argv, cwd, timeout_seconds)` for a single allowlisted executable.

File tools resolve symlinks and reject paths outside `AGENT_WORKSPACE_ROOT`.
`run_command` does not invoke a shell, so pipes, redirection, shell built-ins,
and command chaining are unavailable. It caps runtime and captured output and
passes a reduced environment that excludes API keys. A permitted executable
can still access whatever the operating-system user can access, so the CLI tool
is not an OS sandbox; for untrusted users, run the entire server in a container
or restricted service account and leave `ENABLE_LOCAL_CLI=0`.

## License

MIT
