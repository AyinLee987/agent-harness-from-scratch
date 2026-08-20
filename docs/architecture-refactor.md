# Agent Harness Architecture Refactor

**Date:** 2026-08-12
**Status:** In Progress

## Design: Trigger Layer + State Layer Separation

Split the agent into two clear layers:

### Trigger Layer (`agent/trigger/`)
Decides **when** and **how** the agent acts.

| File | Responsibility |
|---|---|
| `gateway.py` | Unified entry: rate limiting, concurrency control, request queuing |
| `graph.py` | Generic StateGraph engine (pattern-agnostic, moved from `agent/graph.py`) |
| `react_loop.py` | ReAct-specific loop: `think_node()`, `act_node()`, `run()` (extracted from `agent/agent.py`) |
| `dispatch.py` | Tool dispatch: lookup → execute → retry → error handling (extracted from `agent/agent.py`) |

### State Layer (`agent/state/`)
Manages **what** the agent knows.

| File | Responsibility |
|---|---|
| `context.py` | ExecutionContext, Step, budget guard (moved from `agent/context.py`) |
| `memory.py` | ShortTermMemory (window + summarize) + LongTermMemory (vector recall) (moved from `agent/memory.py`) |
| `store.py` | BaseVectorStore + NumPyVectorStore + SQLiteVectorStore (moved from `agent/vector_store.py`) |

### Shared Infrastructure (`agent/`)
Used by both layers.

| File | Responsibility |
|---|---|
| `tools.py` | Tool abstraction, `@tool` decorator, ToolRegistry |
| `llm.py` | LLM clients: MockLLM, OpenAILLM, DeepSeekLLM, BailianLLM |
| `compression.py` | ContextCompressor: query-aware extractive compression |
| `safety.py` | ToolOutputGuard: indirect prompt-injection defense |
| `agent.py` | Thin facade: wires trigger + state + tools + llm together |

## Request Flow

```
Entry (server.py / CLI / Cron)
  → Gateway (rate limit → concurrency → queue)
    → State: ExecutionContext created
    → State: Memory management (short-term + long-term recall)
    → Shared: Optional context compression
    → Trigger: THINK (LLM call)
    → Trigger: Route (tool_calls? → ACT : → answer)
    → Trigger: ACT (dispatch tool → retry)
    → Shared: Safety check (injection guard)
    → State: Budget check + step recording
    → Loop back to THINK or output AgentResult
```

## Directory Structure

```
agent/
├── trigger/
│   ├── __init__.py
│   ├── gateway.py
│   ├── graph.py
│   ├── react_loop.py
│   └── dispatch.py
├── state/
│   ├── __init__.py
│   ├── context.py
│   ├── memory.py
│   └── store.py
├── tools.py
├── llm.py
├── compression.py
├── safety.py
├── agent.py
├── __init__.py
└── eval/
    ├── __init__.py
    ├── harness.py
    └── tasks.json
```
