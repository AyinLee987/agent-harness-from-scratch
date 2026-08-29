"""FastAPI server for the ReAct agent — sync + SSE streaming endpoints.

Start with::

    uvicorn server:app --reload --host 0.0.0.0 --port 8000

Endpoints
---------
``POST /api/run``         —  non-streaming agent run, returns full result as JSON.
``GET  /api/stream``      —  SSE endpoint: streams think/tool_call/tool_result/answer
                             events in real time.  Pass ``task`` as a query param.
``GET  /api/health``      —  health check with model info.
``GET  /api/tools``       —  list registered tools (schemas).
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path as _Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agent import (
    AgentRegistry,
    AgentSpec,
    AgentResult,
    DeepSeekLLM,
    FatalToolError,
    MockLLM,
    MultiAgentOrchestrator,
    ReActAgent,
    RunBudget,
    ToolOutputGuard,
    ToolRegistry,
    ToolDispatcher,
    tool,
)

# ---------------------------------------------------------------------------
# Tools (same three as examples/basic_tools.py)
# ---------------------------------------------------------------------------
import ast as _ast
import operator as _operator
from datetime import datetime as _dt, timezone as _tz

_SAFE_OPS = {
    _ast.Add: _operator.add,
    _ast.Sub: _operator.sub,
    _ast.Mult: _operator.mul,
    _ast.Div: _operator.truediv,
    _ast.Pow: _operator.pow,
    _ast.USub: _operator.neg,
    _ast.UAdd: _operator.pos,
}

def _safe_eval(node):
    if isinstance(node, _ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, _ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, _ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, _ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


_SEARCH_KB = {
    "capital of france": "Paris is the capital of France.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "tallest mountain": "Mount Everest is the tallest mountain on Earth at 8,849 m.",
    "speed of light": "The speed of light is approximately 299,792 km/s.",
    "creator of python": "Python was created by Guido van Rossum.",
}


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression and return the result.
    Args:
        expression: An arithmetic expression, e.g. '23 * 17'.
    """
    try:
        tree = _ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
    except Exception as exc:
        return f"Could not evaluate '{expression}': {exc}"
    if result == int(result):
        return str(int(result))
    return str(result)


@tool
def web_search(query: str) -> str:
    """Look up a fact from a small canned knowledge base.
    Args:
        query: The search query.
    """
    q = query.lower()
    for key, value in _SEARCH_KB.items():
        if key in q:
            return value
    return f"No results found for '{query}'."


@tool
def datetime_now() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_registry() -> ToolRegistry:
    registry = ToolRegistry([calculator, web_search])
    datetime_now.name = "datetime"
    registry.register(datetime_now)
    return registry


# ---------------------------------------------------------------------------
# Build the agent (LLM selection)
# ---------------------------------------------------------------------------
REGISTRY = build_registry()
MCP_MANAGER = None
OUTPUT_GUARD = ToolOutputGuard()

def _build_llm():
    """Pick LLM: DeepSeek > OpenAI > MockLLM fallback."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekLLM()
    if os.getenv("OPENAI_API_KEY") and os.getenv("USE_OPENAI"):
        from agent import OpenAILLM
        return OpenAILLM()
    return MockLLM()


def _fetch_mcp_enabled() -> bool:
    return os.getenv("ENABLE_FETCH_MCP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


LEADER_SYSTEM_PROMPT = (
    "You are the Leader. Solve ordinary tasks directly with the normal tools. "
    "When a task is genuinely separable or benefits from specialist work, use "
    "spawn_subagent with the researcher or analyst role. Start independent "
    "subtasks before calling wait_subagents so they can run in parallel. Read "
    "the Worker results, recover from child failures when possible, and return "
    "one coherent final answer. Do not delegate simple tasks unnecessarily."
)


def _copy_tools(*names: str) -> ToolRegistry:
    selected = [REGISTRY.get(name) for name in names]
    return ToolRegistry([item for item in selected if item is not None])


def _build_leader_runtime(max_steps: int = 10):
    """Create a request-scoped Leader whose delegation ability is a tool set."""

    workers = AgentRegistry()

    def build_researcher() -> ReActAgent:
        names = ["web_search"]
        if "fetch__fetch" in REGISTRY:
            names.append("fetch__fetch")
        return ReActAgent(
            llm=_build_llm(),
            tools=_copy_tools(*names),
            system_prompt=(
                "You are a research Worker. Investigate only the delegated task, "
                "use available source tools when useful, and return a concise "
                "evidence-focused report to the Leader."
            ),
            output_guard=OUTPUT_GUARD,
            max_steps=max_steps,
        )

    def build_analyst() -> ReActAgent:
        return ReActAgent(
            llm=_build_llm(),
            tools=_copy_tools("calculator", "datetime"),
            system_prompt=(
                "You are an analysis Worker. Solve only the delegated task, "
                "return a focused report, and do not delegate further."
            ),
            output_guard=OUTPUT_GUARD,
            max_steps=max_steps,
        )

    workers.register(
        AgentSpec("researcher", "Researches web pages and source material."),
        build_researcher,
    )
    workers.register(
        AgentSpec("analyst", "Performs focused analysis and calculations."),
        build_analyst,
    )
    orchestrator = MultiAgentOrchestrator(
        workers,
        RunBudget(max_subagents=6, max_parallel_tasks=3),
    )
    leader_registry = ToolRegistry(
        [REGISTRY.get(name) for name in REGISTRY.names() if REGISTRY.get(name)]
    )
    leader_registry.register_many(orchestrator.leader_tools())
    leader = ReActAgent(
        llm=_build_llm(),
        tools=leader_registry,
        system_prompt=LEADER_SYSTEM_PROMPT,
        output_guard=OUTPUT_GUARD,
        max_steps=max_steps,
    )
    return orchestrator, leader


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Connect explicitly enabled MCP servers and close them on shutdown."""

    global MCP_MANAGER
    if not _fetch_mcp_enabled():
        yield
        return

    try:
        from agent.mcp import MCPManager, MCPServerConfig, uv_tool_command
    except ModuleNotFoundError as exc:
        if exc.name == "mcp":
            raise RuntimeError(
                "Fetch MCP is enabled but the MCP SDK is missing. "
                "Install dependencies with: python -m pip install -r requirements.txt"
            ) from exc
        raise

    command, args = uv_tool_command("mcp-server-fetch")
    fetch = MCPServerConfig.stdio(
        name="fetch",
        command=command,
        args=args,
        env={"PYTHONIOENCODING": "utf-8"},
        connect_timeout=float(os.getenv("MCP_FETCH_CONNECT_TIMEOUT", "90")),
        call_timeout=float(os.getenv("MCP_FETCH_CALL_TIMEOUT", "60")),
    )
    manager = MCPManager([fetch])
    try:
        tools = await asyncio.to_thread(manager.connect_all)
        REGISTRY.register_many(tools)
        MCP_MANAGER = manager
        yield
    finally:
        await asyncio.to_thread(manager.close)
        MCP_MANAGER = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agent Harness API",
    description="ReAct agent with SSE streaming — no LangChain, pure Python.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Pydantic models -------------------------------------------------------
class RunRequest(BaseModel):
    task: str
    max_steps: int = 10


class RunResponse(BaseModel):
    answer: str
    success: bool
    steps: int
    tokens: int
    stop_reason: str
    trajectory: list[dict[str, Any]]

    @classmethod
    def from_result(cls, r: AgentResult) -> "RunResponse":
        return cls(
            answer=r.answer,
            success=r.success,
            steps=r.steps,
            tokens=r.tokens,
            stop_reason=r.stop_reason,
            trajectory=r.trajectory,
        )


# -- Endpoints -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the Agent Playground UI."""
    playground = _Path(__file__).parent / "web" / "playground.html"
    if not playground.exists():
        raise HTTPException(404, "playground.html not found")
    return HTMLResponse(playground.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    llm = _build_llm()
    return {
        "status": "ok",
        "model": getattr(llm, "model", "mock"),
        "tools": len(REGISTRY),
    }


@app.get("/api/tools")
async def list_tools():
    orchestrator, leader = _build_leader_runtime()
    try:
        return {"tools": leader.tools.schemas()}
    finally:
        orchestrator.close()


@app.post("/api/run", response_model=RunResponse)
async def run(req: RunRequest):
    """Run the request through a Leader with optional delegation tools."""

    def execute():
        orchestrator, leader = _build_leader_runtime(req.max_steps)
        with orchestrator:
            return orchestrator.run_leader(leader, req.task)

    result = await asyncio.to_thread(execute)
    return RunResponse.from_result(result)


@app.get("/api/stream")
async def stream(task: str = Query(..., description="Task for the agent")):
    """SSE streaming endpoint — real-time think / tool_call / tool_result / answer."""

    orchestrator, agent = _build_leader_runtime(max_steps=10)
    llm = agent.llm
    leader_registry = agent.tools

    async def event_stream() -> AsyncGenerator[str, None]:
        from agent.state.context import ExecutionContext

        ctx = ExecutionContext(max_steps=agent.max_steps, max_tokens=agent.max_tokens)
        ctx.add_message("system", agent.system_prompt)
        ctx.add_message("user", task)
        dispatcher = ToolDispatcher(leader_registry)
        stop_reason = "max_steps"
        root_run_id = ""

        try:
            with orchestrator.leader_scope() as root_run_id:
                yield _sse("start", {
                    "task": task,
                    "tools": leader_registry.names(),
                    "root_run_id": root_run_id,
                })

                async for payload in _stream_leader_steps(
                    task=task,
                    agent=agent,
                    llm=llm,
                    registry=leader_registry,
                    dispatcher=dispatcher,
                    ctx=ctx,
                ):
                    if payload[0] == "__stop__":
                        stop_reason = payload[1]["stop_reason"]
                    else:
                        yield _sse(payload[0], payload[1])

            subagents = [
                item.to_dict() for item in orchestrator.results_for_run(root_run_id)
            ]
            yield _sse("done", {
                "steps": len(ctx.steps),
                "tokens": ctx.tokens_used + sum(item["tokens"] for item in subagents),
                "success": stop_reason == "finished",
                "stop_reason": stop_reason,
                "root_run_id": root_run_id,
                "subagents": subagents,
            })
        finally:
            await asyncio.to_thread(orchestrator.close)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_leader_steps(*, task, agent, llm, registry, dispatcher, ctx):
    """Yield the existing streaming ReAct events for one active Leader scope."""

    stop_reason = "max_steps"
    for step_idx in range(agent.max_steps):
        if ctx.over_budget():
            stop_reason = "budget"
            yield "error", {"message": f"Budget exceeded: {ctx.budget_reason()}"}
            break

        # --- THINK phase with streaming ---
        yield "think_start", {"step": step_idx}
        full_content = ""
        tool_calls = []

        async for event in llm.astream(
            agent.short_term.manage(ctx.messages),
            tools=registry.schemas(),
        ):
            if event["type"] == "text":
                full_content += event["data"]
                yield "text", {"step": step_idx, "token": event["data"]}
            elif event["type"] == "tool_call":
                tool_calls.append(event["data"])
                yield "tool_call", {"step": step_idx, "tool": event["data"]}

        ctx.add_tokens(estimate_tokens_simple(full_content))

        # No tool calls → final answer.
        if not tool_calls:
            answer = full_content.strip()
            ctx.add_message("assistant", answer)
            stop_reason = "finished"
            yield "answer", {"step": step_idx, "text": answer}
            break

        # --- ACT phase ---
        yield "act_start", {
            "step": step_idx,
            "tools": [item["name"] for item in tool_calls],
        }

        from agent.llm import ToolCall as TCT

        tc_objects = [
            TCT(id=item["id"], name=item["name"], arguments=item["arguments"])
            for item in tool_calls
        ]
        ctx.add_message(
            "assistant",
            full_content,
            tool_calls=[
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in tc_objects
            ],
        )

        fatal_error = None
        for tc in tc_objects:
            try:
                result = dispatcher.dispatch(ctx, tc.name, tc.arguments)
            except FatalToolError as exc:
                fatal_error = str(exc)
                stop_reason = "fatal_tool_error"
                yield "error", {
                    "step": step_idx,
                    "type": "fatal_tool_error",
                    "message": fatal_error,
                }
                break
            scan = OUTPUT_GUARD.scan(result)
            if scan.suspicious:
                result = scan.sanitized
            ctx.add_message("tool", result, tool_call_id=tc.id, name=tc.name)
            yield "tool_result", {
                "step": step_idx,
                "tool": tc.name,
                "result": result,
            }
        if fatal_error is not None:
            break

    yield "__stop__", {"stop_reason": stop_reason}


# -- helpers ---------------------------------------------------------------
def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def estimate_tokens_simple(text: str) -> int:
    return max(1, len(text or "") // 4)


# ---------------------------------------------------------------------------
# Main (for ``python server.py``)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
