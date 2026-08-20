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
from typing import Any, AsyncGenerator

from dotenv import load_dotenv

load_dotenv()

from pathlib import Path as _Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from agent import (
    AgentResult,
    DeepSeekLLM,
    MockLLM,
    ReActAgent,
    ToolRegistry,
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

def _build_llm():
    """Pick LLM: DeepSeek > OpenAI > MockLLM fallback."""
    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekLLM()
    if os.getenv("OPENAI_API_KEY") and os.getenv("USE_OPENAI"):
        from agent import OpenAILLM
        return OpenAILLM()
    return MockLLM()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Agent Harness API",
    description="ReAct agent with SSE streaming — no LangChain, pure Python.",
    version="0.1.0",
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
    return {"tools": REGISTRY.schemas()}


@app.post("/api/run", response_model=RunResponse)
async def run(req: RunRequest):
    """Non-streaming agent run — submit task, get full result."""
    llm = _build_llm()
    agent = ReActAgent(llm=llm, tools=REGISTRY, max_steps=req.max_steps)

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, agent.run, req.task)
    return RunResponse.from_result(result)


@app.get("/api/stream")
async def stream(task: str = Query(..., description="Task for the agent")):
    """SSE streaming endpoint — real-time think / tool_call / tool_result / answer."""

    llm = _build_llm()
    agent = ReActAgent(llm=llm, tools=REGISTRY, max_steps=10)

    async def event_stream() -> AsyncGenerator[str, None]:
        from agent.state.context import ExecutionContext

        ctx = ExecutionContext(max_steps=agent.max_steps, max_tokens=agent.max_tokens)
        ctx.add_message("system", agent.system_prompt)
        ctx.add_message("user", task)

        yield _sse("start", {"task": task, "tools": REGISTRY.names()})

        for step_idx in range(agent.max_steps):
            if ctx.over_budget():
                yield _sse("error", {"message": f"Budget exceeded: {ctx.budget_reason()}"})
                break

            # --- THINK phase with streaming ---
            yield _sse("think_start", {"step": step_idx})
            full_content = ""
            tool_calls = []

            async for event in llm.astream(
                agent.short_term.manage(ctx.messages),
                tools=REGISTRY.schemas(),
            ):
                if event["type"] == "text":
                    full_content += event["data"]
                    yield _sse("text", {"step": step_idx, "token": event["data"]})
                elif event["type"] == "tool_call":
                    tool_calls.append(event["data"])
                    yield _sse("tool_call", {"step": step_idx, "tool": event["data"]})

            ctx.add_tokens(estimate_tokens_simple(full_content))

            # No tool calls → final answer.
            if not tool_calls:
                answer = full_content.strip()
                ctx.add_message("assistant", answer)
                yield _sse("answer", {"step": step_idx, "text": answer})
                break

            # --- ACT phase ---
            yield _sse("act_start", {"step": step_idx, "tools": [t["name"] for t in tool_calls]})

            from agent.llm import ToolCall as TCT
            tc_objects = [TCT(id=t["id"], name=t["name"], arguments=t["arguments"]) for t in tool_calls]

            ctx.add_message("assistant", full_content, tool_calls=[
                {"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}
                for tc in tc_objects
            ])

            for tc in tc_objects:
                try:
                    result = REGISTRY.dispatch(tc.name, tc.arguments)
                except Exception as exc:
                    result = f"ERROR: {exc}"
                ctx.add_message("tool", result, tool_call_id=tc.id, name=tc.name)
                yield _sse("tool_result", {
                    "step": step_idx,
                    "tool": tc.name,
                    "result": result,
                })

        # --- trajectory ---
        yield _sse("done", {"steps": len(ctx.steps), "tokens": ctx.tokens_used})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
