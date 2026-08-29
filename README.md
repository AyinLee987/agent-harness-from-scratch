# agent-harness-from-scratch

A minimal, production-shaped **ReAct agent framework** built from first principles in pure Python — no LangChain, no LlamaIndex. The goal is to show the *internals* of an agent runtime: the think–act loop, an execution context, a typed tool abstraction, memory, guardrails, and an evaluation harness.

[![CI](https://github.com/sudhanshu-shivam-dev/agent-harness-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/sudhanshu-shivam-dev/agent-harness-from-scratch/actions/workflows/ci.yml)

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
- **Memory** — short-term context management (window + summarization) and long-term vector recall
- **Context compression** — query-aware compression that shrinks the input before the LLM call, inspired by recent long-context research
- **Guardrails** — max-step limits, finish/confidence checks, and safe handling of malformed tool calls
- **Tool-output safety** — an injection guard that screens untrusted tool output for indirect prompt injection
- **Observability** — a run id and per-step correlation ids thread through `ExecutionContext`
- **Evaluation harness** — runs the agent over a task set, logs full trajectories, and scores with rule-based, **trajectory-level**, and optional LLM-as-judge

It runs **with zero setup and no API key** thanks to a deterministic `MockLLM`, and switches to a real model when `OPENAI_API_KEY` is provided.

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

**How memory is layered.** Short-term memory keeps a sliding window of recent
messages and summarizes the overflow so the prompt stays inside the model's
budget. Long-term memory is a separate concern: an embedding store that recalls
relevant facts from *past* runs and primes the system prompt. Splitting them
mirrors how production agents separate "what's in this conversation" from "what
we've learned over time."

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
run concurrently; replace the in-memory NumPy store with a real vector DB
(FAISS/Qdrant/pgvector) and persist it; swap the extractive compressor for a
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
  trigger/           ← Trigger Layer (when / how)
    gateway.py       🆕 Unified entry: rate limiting + concurrency + queuing
    graph.py         Generic StateGraph engine (pattern-agnostic)
    react_loop.py    🆕 ReAct think→act→observe cycle (extracted from agent.py)
    dispatch.py      🆕 Tool execution with retry logic
  state/             ← State Layer (what)
    context.py       ExecutionContext: messages, steps, budget, run_id
    memory.py        ShortTermMemory + LongTermMemory (vector recall)
    store.py         BaseVectorStore → NumPy / SQLite / (FAISS / Qdrant future)
  tools.py           BaseTool + @tool decorator (auto JSON schema) + ToolRegistry
  mcp/               persistent MCP clients + dynamic BaseTool proxies
  llm.py             LLM clients: MockLLM, OpenAI, DeepSeek, Bailian
  compression.py     ContextCompressor: query-aware context compression
  safety.py          ToolOutputGuard: indirect prompt-injection defense
  agent.py           Thin facade: wires trigger + state + shared infrastructure
  eval/
    harness.py       runs tasks; rule-based + trajectory + LLM-as-judge scoring
    tasks.json       sample eval tasks with expected outcomes
examples/
  basic_tools.py         calculator + web-search-stub + datetime tools
  context_compression.py query-aware context compression demo
  prompt_injection.py    tool-output injection guard before/after demo
  mcp_fetch.py            official Fetch MCP registered as fetch__fetch
  memory_demo.py         long-term vector memory recall demo
  run_eval.py            runs the eval harness and prints a scorecard
tests/
  test_agent.py          unit + integration tests (run entirely on MockLLM)
  test_mcp.py            MCP discovery, proxy, lifecycle, and result tests
  test_compression.py    tests for the context compressor
  test_safety.py         tests for the prompt-injection guard
web/
  ReAct Agent Playground — interactive browser demo (Vite + React + TS)
```

## Roadmap

- [x] Trajectory-level eval scoring + judge/rule calibration
- [x] Indirect prompt-injection defense for tool output
- [x] Correlation ids through the execution context
- [x] Trigger / State layer architecture separation
- [x] Gateway with rate limiting + concurrency control
- [ ] Async multi-agent orchestration (planner/executor)
- [x] MCP tool integration (stdio + Streamable HTTP)
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

## License

MIT
