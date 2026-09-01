"""FastAPI server for the ReAct agent — sync + SSE streaming endpoints.

Start with (from the repo root)::

    uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

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
import hmac
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path as _Path
from typing import Any, AsyncGenerator, List, Optional, Sequence

# This module lives at <repo root>/app/server.py — one level below the repo
# root, where the `agent` package lives. Put the repo root on sys.path before
# anything below imports `agent`, so this works whether it's launched as
# `uvicorn app.server:app`, `python app/server.py`, or a debugger pointed
# straight at this file (none of which reliably put the repo root there on
# their own the way `python -m app.server` would).
_REPO_ROOT = _Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# Only needed for the opt-in tool_scaling_kit import in build_registry() below
# (bare `tool_scaling_kit` / `tool_scaling_verbose_kit`, matching how the
# examples/tool_scaling_*.py scripts import each other) — harmless to add
# unconditionally since it's just a sys.path entry, not an import.
_EXAMPLES_DIR = str(_REPO_ROOT / "examples")
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from pydantic import Field

from agent import (
    AgentRegistry,
    AgentSpec,
    AgentResult,
    ContextProvider,
    DeepSeekLLM,
    ExplicitRequestMemoryExtractor,
    FatalToolError,
    MemoryManager,
    MemoryStatus,
    MockLLM,
    MultiAgentOrchestrator,
    LocalToolConfig,
    BM25Retriever,
    CitationCounter,
    DenseRetriever,
    LLMQueryDecomposer,
    LLMReranker,
    MedicalParentChildChunker,
    OpenAICompatibleEmbeddingProvider,
    InMemorySessionStore,
    RAGContextProvider,
    RAGIngestionService,
    RAGPipeline,
    SessionContextProvider,
    SessionMemoryStore,
    SQLiteMemoryRepository,
    SQLiteRAGRepository,
    SQLiteSessionStore,
    create_rag_search_tool,
    create_local_tools,
    ReActAgent,
    RunBudget,
    ToolOutputGuard,
    ToolRegistry,
    ToolDispatcher,
    FORCED_REFLECTION_PROMPT,
    REFLECT_AFTER_FAILURE_STATE_KEY,
    is_failure_observation,
    RequestLoggingMiddleware,
    bind_log_context,
    configure_logging,
    get_logger,
    log_event,
    tool,
)
from app.config import load_agent_config

configure_logging()
logger = get_logger("server")

# Leader/Worker run-tuning knobs (step/token budgets, delegation limits,
# retry/loop-detection behavior, session windowing) -- see config/agent.yaml
# and app/config.py. Loaded once at import time; edit the YAML and restart
# to pick up changes. AGENT_CONFIG_PATH points at a different file.
CONFIG = load_agent_config()

# ---------------------------------------------------------------------------
# Tools
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
def datetime_now() -> str:
    """Return the current UTC date and time in ISO-8601 format."""
    return _dt.now(_tz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_registry() -> ToolRegistry:
    registry = ToolRegistry([calculator])
    datetime_now.name = "datetime"
    registry.register(datetime_now)
    file_tools_enabled = os.getenv("ENABLE_LOCAL_FILE_TOOLS", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    cli_enabled = os.getenv("ENABLE_LOCAL_CLI", "0").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if file_tools_enabled or cli_enabled:
        allowed_commands = tuple(
            item.strip() for item in os.getenv(
                "AGENT_CLI_ALLOWED_COMMANDS", "git,rg,python,python.exe,pytest"
            ).split(",") if item.strip()
        )
        config = LocalToolConfig(
            workspace_root=os.getenv("AGENT_WORKSPACE_ROOT", str(_REPO_ROOT)),
            max_read_bytes=int(os.getenv("AGENT_FILE_MAX_READ_BYTES", str(2 * 1024 * 1024))),
            max_write_bytes=int(os.getenv("AGENT_FILE_MAX_WRITE_BYTES", str(2 * 1024 * 1024))),
            max_command_output_bytes=int(os.getenv("AGENT_CLI_MAX_OUTPUT_BYTES", str(256 * 1024))),
            command_timeout_seconds=float(os.getenv("AGENT_CLI_TIMEOUT_SECONDS", "30")),
            allowed_commands=allowed_commands,
        )
        registry.register_many(create_local_tools(
            config, include_files=file_tools_enabled, include_cli=cli_enabled,
        ))
        log_event(
            logger,
            logging.INFO,
            "local_tools.registered",
            workspace_root=str(config.workspace_root),
            file_tools_enabled=file_tools_enabled,
            cli_enabled=cli_enabled,
            allowed_command_count=len(allowed_commands),
        )
    _raw_scaling_flag = os.getenv("ENABLE_TOOL_SCALING_KIT", "0")
    scaling_kit_enabled = _raw_scaling_flag.strip().lower() in {
        "1", "true", "yes", "on",
    }
    # Always log this decision (not just when enabled) -- the raw value is
    # included so a stale process env (e.g. a debugger session started
    # before an .env edit, or a launcher that pre-populates os.environ from
    # its own cached copy) is visible in the logs instead of silently
    # looking like the kit was never wired in at all.
    log_event(
        logger,
        logging.INFO,
        "tool_scaling_kit.flag_read",
        raw_value=_raw_scaling_flag,
        enabled=scaling_kit_enabled,
    )
    if scaling_kit_enabled:
        verbose = os.getenv("TOOL_SCALING_KIT_VERBOSE", "0").strip().lower() in {
            "1", "true", "yes", "on",
        }
        try:
            size = int(os.getenv("TOOL_SCALING_KIT_SIZE", "50"))
        except ValueError:
            size = 50
        if verbose:
            from tool_scaling_verbose_kit import ALL_TOOLS_VERBOSE as _kit_tools
        else:
            from tool_scaling_kit import ALL_TOOLS as _kit_tools
        size = max(1, min(size, len(_kit_tools)))
        registry.register_many(_kit_tools[:size])
        log_event(
            logger,
            logging.INFO,
            "tool_scaling_kit.registered",
            verbose=verbose,
            tool_count=size,
        )
    return registry


# ---------------------------------------------------------------------------
# Build the agent (LLM selection)
# ---------------------------------------------------------------------------
REGISTRY = build_registry()
MCP_MANAGER = None
RAG_REPOSITORY = None
RAG_PIPELINE = None
RAG_INGESTION = None
MEMORY_MANAGER: Optional[MemoryManager] = None
# There's no auth/user system in this server -- every request is a single
# local deployment, distinguished only by its (optional) conversation_id.
# So long-term memory uses one fixed namespace/subject for everyone: a
# "remember ..." in one conversation is recallable from any other
# conversation, which is the point (persistent facts about *the* user, not
# a conversation-scoped fact). If this server ever grows real accounts,
# memory_subject_id should become per-authenticated-user instead.
MEMORY_NAMESPACE = "default"
MEMORY_SUBJECT_ID = "web-ui"
OUTPUT_GUARD = ToolOutputGuard()
# Swapped for a SQLiteSessionStore in lifespan() startup; kept as a
# dependency-free default here so importing this module (e.g. from a test
# that calls route functions directly, bypassing lifespan) never touches
# disk — see AGENTS.md's rule on production defaulting to durable, tests to
# in-memory.
SESSION_STORE: SessionMemoryStore = InMemorySessionStore()

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


def _enabled(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _fetch_server_args(base_args: List[str]) -> List[str]:
    """Append ``--ignore-robots-txt`` when explicitly opted in.

    mcp-server-fetch respects robots.txt by default, which most major
    sites (LinkedIn, Facebook, Reddit, Twitter/X, ...) use to explicitly
    disallow automated fetching -- verified live: fetching any of those
    fails with an explicit "robots.txt ... specifies that autonomous
    fetching ... is not allowed" error, which is most of why a research
    Worker's fetch calls fail so often in practice. This flag makes the
    tool ignore robots.txt and fetch anyway. It is a deliberate opt-in past
    sites' stated access policy, not a bug fix -- off by default; see
    .env.example and the README for the tradeoff before turning it on.
    """
    if _enabled("MCP_FETCH_IGNORE_ROBOTS_TXT"):
        return [*base_args, "--ignore-robots-txt"]
    return list(base_args)


def _start_rag():
    """Build the governed RAG runtime and ingest configured text sources."""
    global RAG_REPOSITORY, RAG_PIPELINE, RAG_INGESTION
    if not _enabled("ENABLE_RAG"):
        return
    model = os.getenv("RAG_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBED_MODEL")
    if not model:
        raise RuntimeError("ENABLE_RAG=1 requires RAG_EMBEDDING_MODEL.")
    api_key = os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    repository_path = _Path(os.getenv("RAG_DB_PATH", "data/rag.sqlite"))
    repository_path.parent.mkdir(parents=True, exist_ok=True)
    repository = SQLiteRAGRepository(repository_path)
    bm25 = BM25Retriever(repository)
    dense = DenseRetriever(repository, OpenAICompatibleEmbeddingProvider(
        model=model, api_key=api_key, base_url=base_url, provider_name="rag",
    ))
    bm25.rebuild()
    dense.rebuild()
    ingestion = RAGIngestionService(
        repository, MedicalParentChildChunker(), [bm25, dense]
    )
    source_dir = _Path(os.getenv("RAG_SOURCE_DIR", "data/rag_sources"))
    if source_dir.exists():
        for path in sorted(source_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
                ingestion.ingest_text(
                    logical_id=path.relative_to(source_dir).as_posix(),
                    title=path.stem,
                    content=path.read_text(encoding="utf-8"),
                    source_url=path.resolve().as_uri(),
                    publisher=os.getenv("RAG_DEFAULT_PUBLISHER", "local-corpus"),
                    jurisdiction=os.getenv("RAG_DEFAULT_JURISDICTION", "CN"),
                )
    # Opt-in: classifies each question (single_hop/parallel/sequential) and
    # retrieves independent sub-questions separately before merging -- see
    # agent/rag/decomposition.py. Off by default: one extra LLM call per
    # retrieval, only worth it if your corpus actually has compound/
    # multi-hop questions to answer.
    decomposer = LLMQueryDecomposer(_build_llm()) if _enabled("ENABLE_RAG_QUERY_DECOMPOSITION") else None
    # Opt-in: replace the default HeuristicReranker (lexical-overlap
    # heuristic) with one that asks the chat model itself to score each
    # candidate's relevance. Real cost: ~2.2s added per retrieval (one more
    # LLM call) -- measured on BEIR NFCorpus (see the companion evaluation
    # project's benchmarks/rag_recall_beir/RESULTS_llm_rerank.md) to beat
    # both plain RRF fusion and HeuristicReranker on every metric,
    # significantly on Recall@5/10 and nDCG@5/10 -- not on MRR. Off by
    # default because of that latency, not because it measures worse.
    reranker = LLMReranker(_build_llm()) if _enabled("ENABLE_LLM_RERANKER") else None
    RAG_REPOSITORY = repository
    RAG_PIPELINE = RAGPipeline(repository, bm25, dense, reranker=reranker, decomposer=decomposer)
    RAG_INGESTION = ingestion
    log_event(
        logger, logging.INFO, "rag.started",
        document_count=len(repository.list_documents()),
        query_decomposition_enabled=decomposer is not None,
        llm_reranker_enabled=reranker is not None,
    )


def _start_memory():
    """Build the policy-controlled long-term memory runtime, if enabled.

    Off by default. When on, the Leader gets a ``memory_search`` tool (model
    decides when to recall, same philosophy as RAG's search tool) and every
    successful run is passed through ``ExplicitRequestMemoryExtractor`` +
    ``DefaultMemoryPolicy`` -- only an explicit "remember ..."/"记住..."
    message ever gets persisted, nothing is inferred from ordinary
    conversation. See ``agent/memory/policy.py``.
    """
    global MEMORY_MANAGER
    if not _enabled("ENABLE_LONG_TERM_MEMORY"):
        return
    model = (
        os.getenv("MEMORY_EMBEDDING_MODEL")
        or os.getenv("RAG_EMBEDDING_MODEL")
        or os.getenv("OPENAI_EMBED_MODEL")
    )
    if not model:
        raise RuntimeError("ENABLE_LONG_TERM_MEMORY=1 requires MEMORY_EMBEDDING_MODEL.")
    api_key = (
        os.getenv("MEMORY_EMBEDDING_API_KEY")
        or os.getenv("RAG_EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    base_url = (
        os.getenv("MEMORY_EMBEDDING_BASE_URL")
        or os.getenv("RAG_EMBEDDING_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    db_path = os.getenv("MEMORY_DB_PATH", "data/memory.sqlite")
    repository = SQLiteMemoryRepository(db_path)
    embeddings = OpenAICompatibleEmbeddingProvider(
        model=model, api_key=api_key, base_url=base_url, provider_name="memory",
    )
    manager = MemoryManager(
        embeddings,
        repository=repository,
        extractor=ExplicitRequestMemoryExtractor(),
    )
    # Records may already exist from a prior process (SQLite persists); the
    # vector index is derived/in-memory and starts empty every boot, so it
    # must be rebuilt from the repository (the source of truth) before any
    # recall can find them.
    restored = manager.rebuild_index()
    MEMORY_MANAGER = manager
    log_event(logger, logging.INFO, "memory.started", restored_record_count=restored)


LEADER_SYSTEM_PROMPT = (
    "You are the Leader. Solve ordinary tasks directly with the normal tools. "
    "When a task is genuinely separable or benefits from specialist work, use "
    "spawn_subagent with the researcher or analyst role. Start independent "
    "subtasks before calling wait_subagents so they can run in parallel. Read "
    "the Worker results, recover from child failures when possible, and return "
    "one coherent final answer. Do not delegate simple tasks unnecessarily. "
    "Once a Worker reports back (even a partial or failed result), use what "
    "it already found — do not redo the same investigation yourself with your "
    "own tools; that wastes your own step budget on work already attempted. "
    "If the Worker's result is genuinely insufficient, say so plainly in your "
    "final answer rather than silently retrying the same approach. When using "
    "fetch yourself, some sites block automated requests -- if a source fails "
    "twice in a row, stop retrying it and try a different one instead of "
    "digging deeper into a source that isn't going to answer."
)


def _copy_tools(*names: str) -> ToolRegistry:
    selected = [REGISTRY.get(name) for name in names]
    return ToolRegistry([item for item in selected if item is not None])


def _build_leader_runtime(
    max_steps: Optional[int] = None,
    extra_context_providers: Optional[Sequence[ContextProvider]] = None,
):
    """Create a request-scoped Leader whose delegation ability is a tool set.

    ``max_steps=None`` (the default) lets the Leader and each Worker use
    their own independent step budget from ``config/agent.yaml``
    (``CONFIG.leader.max_steps`` / ``CONFIG.worker.max_steps``). Passing an
    explicit value (e.g. a client's ``POST /api/run`` request) overrides
    both with that one number, matching the pre-config behavior where a
    single ``max_steps`` applied to the whole run.
    """

    leader_max_steps = CONFIG.leader.max_steps if max_steps is None else max_steps
    worker_max_steps = CONFIG.worker.max_steps if max_steps is None else max_steps

    workers = AgentRegistry()

    def build_researcher() -> ReActAgent:
        names = []
        if "fetch__fetch" in REGISTRY:
            names.append("fetch__fetch")
        worker_tools = _copy_tools(*names)
        if RAG_PIPELINE:
            # Own counter, scoped to this worker instance: a researcher's
            # medical_evidence_search calls stay inside its own private
            # trajectory (only its final text report reaches the Leader),
            # so they never share a message list -- and thus never share a
            # citation-numbering space -- with the Leader's own citations.
            worker_tools.register(create_rag_search_tool(
                RAG_PIPELINE, citation_counter=CitationCounter(),
            ))
        return ReActAgent(
            llm=_build_llm(),
            tools=worker_tools,
            system_prompt=(
                "You are a research Worker. Investigate only the delegated task, "
                "use available source tools when useful, and return a concise "
                "evidence-focused report to the Leader. Some sites block automated "
                "fetching (robots.txt, anti-bot challenges, rate limits) -- if a "
                "source fails twice in a row, stop retrying it and move to a "
                "different source instead. For any one sub-question, two or three "
                "sources are usually enough; report what you found (including gaps) "
                "rather than exhaustively chasing every remaining lead."
            ),
            output_guard=OUTPUT_GUARD,
            max_steps=worker_max_steps,
            max_tool_retries=CONFIG.react_loop.max_tool_retries,
            loop_same_call_limit=CONFIG.react_loop.loop_same_call_limit,
            source_failure_hint_threshold=CONFIG.react_loop.source_failure_hint_threshold,
            compress_at_fraction=CONFIG.react_loop.compress_at_fraction,
            agent_name="researcher",
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
            max_steps=worker_max_steps,
            max_tool_retries=CONFIG.react_loop.max_tool_retries,
            loop_same_call_limit=CONFIG.react_loop.loop_same_call_limit,
            source_failure_hint_threshold=CONFIG.react_loop.source_failure_hint_threshold,
            compress_at_fraction=CONFIG.react_loop.compress_at_fraction,
            agent_name="analyst",
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
        RunBudget(
            max_subagents=CONFIG.run_budget.max_subagents,
            max_parallel_tasks=CONFIG.run_budget.max_parallel_tasks,
            max_depth=CONFIG.run_budget.max_depth,
            max_repeated_task=CONFIG.run_budget.max_repeated_task,
            subagent_timeout_seconds=CONFIG.run_budget.subagent_timeout_seconds,
        ),
    )
    leader_registry = ToolRegistry(
        [REGISTRY.get(name) for name in REGISTRY.names() if REGISTRY.get(name)]
    )
    # One counter shared between the Leader's mandatory RAG injection and
    # its medical_evidence_search tool, scoped to this one run -- see
    # CitationCounter's docstring for the [E1]/[E1] collision this avoids.
    leader_citation_counter = CitationCounter()
    if RAG_PIPELINE:
        leader_registry.register(create_rag_search_tool(
            RAG_PIPELINE, citation_counter=leader_citation_counter,
        ))
    if MEMORY_MANAGER:
        leader_registry.register(MEMORY_MANAGER.as_search_tool(
            namespace=MEMORY_NAMESPACE, subject_id=MEMORY_SUBJECT_ID,
        ))
    leader_registry.register_many(orchestrator.leader_tools())
    context_providers: List[ContextProvider] = []
    if RAG_PIPELINE:
        context_providers.append(RAGContextProvider(
            RAG_PIPELINE, citation_counter=leader_citation_counter,
        ))
    context_providers.extend(extra_context_providers or [])
    leader = ReActAgent(
        llm=_build_llm(),
        tools=leader_registry,
        system_prompt=LEADER_SYSTEM_PROMPT,
        output_guard=OUTPUT_GUARD,
        max_steps=leader_max_steps,
        max_tokens=CONFIG.leader.max_tokens,
        max_tool_retries=CONFIG.react_loop.max_tool_retries,
        loop_same_call_limit=CONFIG.react_loop.loop_same_call_limit,
        source_failure_hint_threshold=CONFIG.react_loop.source_failure_hint_threshold,
        compress_at_fraction=CONFIG.react_loop.compress_at_fraction,
        agent_name="leader",
        context_providers=context_providers or None,
        memory_manager=MEMORY_MANAGER,
        memory_namespace=MEMORY_NAMESPACE,
        memory_subject_id=MEMORY_SUBJECT_ID,
    )
    return orchestrator, leader


def _start_session_store() -> SessionMemoryStore:
    """Swap the module-default in-memory session store for a durable one."""

    return SQLiteSessionStore(os.getenv("SESSION_DB_PATH", "data/sessions.sqlite"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Connect explicitly enabled MCP servers and close them on shutdown."""

    global MCP_MANAGER, RAG_REPOSITORY, RAG_PIPELINE, RAG_INGESTION, MEMORY_MANAGER, SESSION_STORE
    manager = None
    try:
        await asyncio.to_thread(_start_rag)
        await asyncio.to_thread(_start_memory)
        SESSION_STORE = await asyncio.to_thread(_start_session_store)
        if _fetch_mcp_enabled():
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
            args = _fetch_server_args(args)
            fetch = MCPServerConfig.stdio(
                name="fetch", command=command, args=args,
                env={"PYTHONIOENCODING": "utf-8"},
                connect_timeout=float(os.getenv("MCP_FETCH_CONNECT_TIMEOUT", "90")),
                call_timeout=float(os.getenv("MCP_FETCH_CALL_TIMEOUT", "60")),
            )
            manager = MCPManager([fetch])
            tools = await asyncio.to_thread(manager.connect_all)
            REGISTRY.register_many(tools)
            MCP_MANAGER = manager
        yield
    finally:
        if manager is not None:
            await asyncio.to_thread(manager.close)
        MCP_MANAGER = None
        if RAG_REPOSITORY is not None:
            await asyncio.to_thread(RAG_REPOSITORY.close)
        RAG_REPOSITORY = None
        RAG_PIPELINE = None
        RAG_INGESTION = None
        if MEMORY_MANAGER is not None and isinstance(MEMORY_MANAGER.repository, SQLiteMemoryRepository):
            await asyncio.to_thread(MEMORY_MANAGER.repository.close)
        MEMORY_MANAGER = None
        if isinstance(SESSION_STORE, SQLiteSessionStore):
            await asyncio.to_thread(SESSION_STORE.close)
        SESSION_STORE = InMemorySessionStore()


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
app.add_middleware(RequestLoggingMiddleware)


# -- Pydantic models -------------------------------------------------------
class RunRequest(BaseModel):
    task: str
    # None (the default) lets the Leader and each Worker use their own
    # independent step budget from config/agent.yaml -- see
    # _build_leader_runtime. Passing a value overrides both with that one
    # number, same as before this was configurable.
    max_steps: Optional[int] = None
    conversation_id: Optional[str] = Field(
        default=None,
        description="Pass back the conversation_id from a prior response to "
        "continue that conversation. Omit it to start a new one.",
    )


class RAGDocumentRequest(BaseModel):
    logical_id: str = Field(min_length=1, max_length=300)
    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    source_url: str = ""
    publisher: str = "unknown"
    document_type: str = "reference"
    jurisdiction: str = ""
    language: str = "zh-CN"
    version: str = "1"
    published_at: _dt | None = None
    effective_at: _dt | None = None
    reviewed_at: _dt | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RAGDocumentResponse(BaseModel):
    document_id: str
    logical_id: str
    status: str
    version: str
    chunk_count: int
    skipped: bool
    supersedes_id: str | None = None
    warnings: list[str]


class RunResponse(BaseModel):
    answer: str
    success: bool
    steps: int
    tokens: int
    stop_reason: str
    trajectory: list[dict[str, Any]]
    conversation_id: Optional[str] = Field(
        default=None,
        description="Echoes the conversation this run belongs to; pass it "
        "back on the next request to continue. Absent when the caller "
        "didn't ask for continuation.",
    )

    @classmethod
    def from_result(
        cls, r: AgentResult, *, conversation_id: Optional[str] = None
    ) -> "RunResponse":
        return cls(
            answer=r.answer,
            success=r.success,
            steps=r.steps,
            tokens=r.tokens,
            stop_reason=r.stop_reason,
            trajectory=r.trajectory,
            conversation_id=conversation_id,
        )


# -- Endpoints -------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the Agent Playground UI."""
    playground = _REPO_ROOT / "web" / "playground.html"
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


@app.post("/api/rag/documents", response_model=RAGDocumentResponse, status_code=201)
async def ingest_rag_document(payload: RAGDocumentRequest, request: Request):
    """Validate, chunk, index, and atomically publish one corpus document."""
    if RAG_INGESTION is None:
        raise HTTPException(503, "RAG is disabled. Set ENABLE_RAG=1 and restart the server.")
    expected_token = os.getenv("RAG_ADMIN_TOKEN", "")
    supplied_token = request.headers.get("X-RAG-Admin-Token", "")
    if not expected_token:
        raise HTTPException(503, "Corpus writes are disabled until RAG_ADMIN_TOKEN is configured.")
    if not hmac.compare_digest(supplied_token, expected_token):
        raise HTTPException(401, "Invalid corpus administration token.")
    maximum_bytes = int(os.getenv("RAG_MAX_DOCUMENT_BYTES", str(5 * 1024 * 1024)))
    if len(payload.content.encode("utf-8")) > maximum_bytes:
        raise HTTPException(413, f"Document exceeds the {maximum_bytes}-byte ingestion limit.")
    dates = {
        key: value for key, value in {
            "published_at": payload.published_at,
            "effective_at": payload.effective_at,
            "reviewed_at": payload.reviewed_at,
        }.items() if value is not None
    }
    try:
        result = await asyncio.to_thread(
            RAG_INGESTION.ingest_text,
            logical_id=payload.logical_id,
            title=payload.title,
            content=payload.content,
            source_url=payload.source_url,
            publisher=payload.publisher,
            document_type=payload.document_type,
            jurisdiction=payload.jurisdiction,
            language=payload.language,
            version=payload.version,
            metadata=payload.metadata,
            **dates,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        log_event(logger, logging.ERROR, "rag.ingestion.failed", exc_info=True)
        raise HTTPException(503, "Corpus indexing failed; the document was not published.") from exc
    log_event(
        logger, logging.INFO, "rag.ingestion.completed",
        document_id=result.document.id, logical_id=result.document.logical_id,
        chunk_count=len(result.chunks), skipped=result.skipped,
    )
    return RAGDocumentResponse(
        document_id=result.document.id,
        logical_id=result.document.logical_id,
        status=result.document.status.value,
        version=result.document.version,
        chunk_count=len(result.chunks),
        skipped=result.skipped,
        supersedes_id=result.document.supersedes_id,
        warnings=result.warnings,
    )


@app.get("/api/tools")
async def list_tools():
    orchestrator, leader = _build_leader_runtime()
    try:
        return {"tools": leader.tools.schemas()}
    finally:
        orchestrator.close()


@app.get("/api/memory")
async def list_memory():
    """Debug/introspection: every ACTIVE long-term memory record.

    Read-only, no pagination -- meant for inspecting what got persisted
    while testing (e.g. confirming two contradictory "remember ..."
    statements both landed as separate records), not as a production API.
    404s with a clear reason if ENABLE_LONG_TERM_MEMORY isn't set.
    """
    if MEMORY_MANAGER is None:
        raise HTTPException(404, "Long-term memory is not enabled (set ENABLE_LONG_TERM_MEMORY=1).")

    def _list():
        records = MEMORY_MANAGER.repository.list_records(
            namespace=MEMORY_NAMESPACE, subject_id=MEMORY_SUBJECT_ID, status=MemoryStatus.ACTIVE,
        )
        return [
            {
                "id": r.id,
                "content": r.content,
                "kind": r.kind.value,
                "created_at": r.created_at.isoformat(),
                "source_run_id": r.source_run_id,
            }
            for r in sorted(records, key=lambda r: r.created_at)
        ]

    records = await asyncio.to_thread(_list)
    return {"namespace": MEMORY_NAMESPACE, "subject_id": MEMORY_SUBJECT_ID, "records": records}


@app.post("/api/run", response_model=RunResponse)
async def run(req: RunRequest):
    """Run the request through a Leader with optional delegation tools.

    Stateless by default, matching every run before this endpoint knew
    conversations existed. Only when the caller sets ``conversation_id`` —
    a client-generated id, reused across calls to continue a conversation —
    does the run get threaded onto that conversation's stored history via
    :class:`SessionContextProvider`, with the turn persisted to
    ``SESSION_STORE`` afterwards. See ``agent/memory/context.py`` for why a
    stateless ``ReActLoop.run()`` needs this at all.
    """

    extra_providers: List[ContextProvider] = []
    session_provider: Optional[SessionContextProvider] = None
    if req.conversation_id:
        session_provider = SessionContextProvider(
            SESSION_STORE,
            req.conversation_id,
            llm=_build_llm(),
            recent_window=CONFIG.session.recent_window,
            summarize_beyond=CONFIG.session.summarize_beyond,
        )
        extra_providers.append(session_provider)

    def execute():
        orchestrator, leader = _build_leader_runtime(
            req.max_steps, extra_context_providers=extra_providers
        )
        with orchestrator:
            return orchestrator.run_leader(leader, req.task)

    result = await asyncio.to_thread(execute)
    if session_provider is not None:
        await asyncio.to_thread(session_provider.record_turn, req.task, result.answer)
    return RunResponse.from_result(result, conversation_id=req.conversation_id)


@app.get("/api/conversations/{conversation_id}")
async def conversation_messages(conversation_id: str):
    """Return a stored conversation's turns, so a UI can restore it.

    An id nothing has been recorded under yet is not an error — it's just a
    new conversation the client generated but hasn't sent a turn on — so
    this returns an empty list rather than 404ing.
    """

    messages = await asyncio.to_thread(SESSION_STORE.load_messages, conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}


@app.get("/api/conversations/{conversation_id}/title")
async def conversation_title(conversation_id: str):
    """A short display title for a stored conversation — for a sidebar/history UI.

    Nothing here is persisted: ``SESSION_STORE``'s messages are the only
    source of truth, this just describes them. Re-calling this can return a
    differently-worded title each time, since it's a fresh LLM call, not a
    stored value — callers (the playground UI) cache the result client-side
    rather than re-requesting it on every render.
    """

    messages = await asyncio.to_thread(SESSION_STORE.load_messages, conversation_id)
    if not messages:
        raise HTTPException(404, "Unknown or empty conversation_id.")

    first_user = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    fallback = (first_user[:24] + "…") if len(first_user) > 24 else (first_user or "对话")

    llm = _build_llm()
    transcript = "\n".join(
        f"{m.get('role')}: {m.get('content')}" for m in messages[:6] if m.get("content")
    )
    prompt = [
        {
            "role": "system",
            "content": "Write a short title summarizing what this conversation "
            "is about, at most 6 words, in the same language the conversation "
            "is written in. Output only the title itself — no quotes, no "
            "trailing punctuation, no explanation.",
        },
        {"role": "user", "content": transcript},
    ]
    resp = await asyncio.to_thread(llm.chat, prompt, [])
    title = (resp.content or "").strip().strip("\"“”")
    if not title or len(title) > 60:
        title = fallback
    return {"conversation_id": conversation_id, "title": title}


@app.get("/api/stream")
async def stream(
    task: str = Query(..., description="Task for the agent"),
    conversation_id: Optional[str] = None,
):
    """SSE streaming endpoint — real-time think / tool_call / tool_result / answer.

    Supports the same opt-in ``conversation_id`` continuity as ``POST /api/run``
    (see there, and ``agent/memory/context.py``, for why a stateless
    ``ReActLoop`` needs this at all) — stateless by default, threaded onto
    stored history and persisted afterwards only when the caller sets one.
    Passed as a plain query param, e.g. ``/api/stream?task=...&conversation_id=conv-123``.

    Note: unlike ``task``, this is intentionally *not* wrapped in
    ``fastapi.Query(...)`` — a route function called directly (as this
    project's tests do, bypassing FastAPI's request-parsing layer) would
    otherwise receive the unresolved ``Query`` sentinel object instead of
    ``None`` when the caller omits it.
    """

    session_provider: Optional[SessionContextProvider] = None
    if conversation_id:
        session_provider = SessionContextProvider(
            SESSION_STORE,
            conversation_id,
            llm=_build_llm(),
            recent_window=CONFIG.session.recent_window,
            summarize_beyond=CONFIG.session.summarize_beyond,
        )

    orchestrator, agent = _build_leader_runtime(
        extra_context_providers=[session_provider] if session_provider else None,
    )
    llm = agent.llm
    leader_registry = agent.tools

    async def event_stream() -> AsyncGenerator[str, None]:
        from agent.state.context import ExecutionContext

        ctx = ExecutionContext(max_steps=agent.max_steps, max_tokens=agent.max_tokens)
        ctx.add_message("system", agent.system_prompt)
        for provider in agent.context_providers:
            for message in await asyncio.to_thread(provider.prepare, task):
                if message.get("content"):
                    ctx.add_message(str(message.get("role", "system")), str(message["content"]))
        ctx.add_message("user", task)
        dispatcher = ToolDispatcher(
            leader_registry,
            max_retries=CONFIG.react_loop.max_tool_retries,
            source_failure_hint_threshold=CONFIG.react_loop.source_failure_hint_threshold,
        )
        stop_reason = "max_steps"
        root_run_id = ""
        final_answer: Optional[str] = None

        # Persist the question the moment the run starts, not after it
        # finishes — a disconnect (e.g. the client refreshing mid-stream) or
        # a run that never reaches a clean "answer" event (budget exhausted,
        # fatal tool error, cancelled) would otherwise leave nothing in
        # SESSION_STORE for this turn, so the conversation can look like it
        # vanished on the next reload even though it clearly happened.
        if session_provider is not None:
            await asyncio.to_thread(session_provider.record_user_message, task)

        try:
            with orchestrator.leader_scope() as root_run_id:
                with bind_log_context(run_id=ctx.run_id, agent_name="leader"):
                    stream_started = time.perf_counter()
                    log_event(
                        logger,
                        logging.INFO,
                        "agent.stream.started",
                        task_chars=len(task),
                        max_steps=agent.max_steps,
                        tool_count=len(leader_registry),
                    )
                    yield _sse("start", {
                        "task": task,
                        "tools": leader_registry.names(),
                        "root_run_id": root_run_id,
                        "conversation_id": conversation_id,
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
                            if payload[0] == "answer":
                                final_answer = payload[1]["text"]
                            yield _sse(payload[0], payload[1])

                    log_event(
                        logger,
                        logging.INFO,
                        "agent.stream.completed",
                        success=stop_reason == "finished",
                        stop_reason=stop_reason,
                        steps=len(ctx.steps),
                        tokens=ctx.tokens_used,
                        elapsed_ms=round((time.perf_counter() - stream_started) * 1000, 2),
                    )

            # The question was already persisted as soon as the run started
            # (see above). Only a run that actually produced a final answer
            # has anything coherent to add for the assistant's side — a
            # cancelled/budget/fatal-error stop just leaves the question
            # standing alone, which is an honest record of what happened.
            if session_provider is not None and final_answer is not None:
                await asyncio.to_thread(
                    session_provider.record_assistant_message, final_answer
                )

            # include_trajectory=True here (unlike the Leader-facing
            # wait_subagents/get_subagent_status tools, which stay compact
            # to avoid bloating the model's own context): this goes
            # straight to the frontend as SSE data, never back into any
            # LLM prompt, so there's no token cost to giving the UI the
            # full per-step detail -- see SubagentResult.tool_call_summary.
            # result_payloads_for_run (not results_for_run) so each entry
            # carries its final "status" -- a Worker the Leader never polled
            # again after spawning it would otherwise report with no status,
            # and the playground merges this onto the card it already drew
            # from the spawn_subagent result, which *does* have a status
            # ("pending") -- without one here to overwrite it, that card
            # would look stuck on "pending" forever despite success/answer
            # already reflecting the real outcome.
            subagents = orchestrator.result_payloads_for_run(
                root_run_id, include_trajectory=True
            )
            yield _sse("done", {
                "steps": len(ctx.steps),
                "tokens": ctx.tokens_used + sum(item["tokens"] for item in subagents),
                "success": stop_reason == "finished",
                "stop_reason": stop_reason,
                "root_run_id": root_run_id,
                "subagents": subagents,
                "conversation_id": conversation_id,
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

        # A failed tool call in the previous step's ACT phase set this --
        # force this THINK call to run with no tools offered so the model
        # must respond in plain text instead of immediately firing off
        # another tool call. Mirrors ReActLoop._think_node's
        # REFLECT_AFTER_FAILURE_STATE_KEY handling; see that module's
        # docstring for why a soft hint in the observation text wasn't
        # enough on its own.
        reflect = ctx.state.pop(REFLECT_AFTER_FAILURE_STATE_KEY, False)

        # --- THINK phase with streaming ---
        yield "think_start", {"step": step_idx}
        full_content = ""
        tool_calls = []
        llm_started = time.perf_counter()
        log_event(
            logger,
            logging.DEBUG,
            "llm.stream.started",
            step=step_idx,
            message_count=len(ctx.messages),
            tool_count=0 if reflect else len(registry),
            forced_reflection=reflect,
        )

        try:
            async for event in llm.astream(
                agent.short_term.manage(ctx.messages),
                tools=None if reflect else registry.schemas(),
            ):
                if event["type"] == "text":
                    full_content += event["data"]
                    yield "text", {"step": step_idx, "token": event["data"]}
                elif event["type"] == "tool_call" and not reflect:
                    tool_calls.append(event["data"])
                    yield "tool_call", {"step": step_idx, "tool": event["data"]}
        except Exception:
            log_event(
                logger,
                logging.ERROR,
                "llm.stream.failed",
                step=step_idx,
                elapsed_ms=round((time.perf_counter() - llm_started) * 1000, 2),
                exc_info=True,
            )
            raise

        log_event(
            logger,
            logging.INFO,
            "llm.stream.completed",
            step=step_idx,
            tool_call_count=len(tool_calls),
            output_chars=len(full_content),
            elapsed_ms=round((time.perf_counter() - llm_started) * 1000, 2),
        )

        ctx.add_tokens(estimate_tokens_simple(full_content))

        if reflect:
            # No tools were offered, so this can only be a plain-text
            # reasoning turn -- never a final answer. Record it and loop
            # straight back to a normal (tool-enabled) think step.
            reasoning = full_content.strip() or "(no reasoning provided)"
            ctx.add_message("assistant", reasoning)
            yield "reflection", {"step": step_idx, "text": reasoning}
            continue

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
        any_failed = False
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
            if is_failure_observation(result):
                any_failed = True
            ctx.add_message("tool", result, tool_call_id=tc.id, name=tc.name)
            yield "tool_result", {
                "step": step_idx,
                "tool": tc.name,
                "result": result,
            }
        if fatal_error is not None:
            break
        if any_failed:
            # See REFLECT_AFTER_FAILURE_STATE_KEY above: forces the next
            # THINK call to run tool-less.
            ctx.add_message("user", FORCED_REFLECTION_PROMPT)
            ctx.state[REFLECT_AFTER_FAILURE_STATE_KEY] = True
            log_event(
                logger, logging.INFO, "reflection.forced", tool_name=tc_objects[0].name
            )
    else:
        # The for loop ran out of iterations without ever `break`-ing (no
        # "finished"/budget/fatal_tool_error case fired) -- stop_reason is
        # still its initial "max_steps" default. Unlike those other stop
        # paths, this one never yielded an "error" event, so the frontend
        # trace just went quiet with no visible outcome at all -- a run
        # that genuinely ran to its full step budget could look exactly
        # like a dropped connection. See README's Structured logging /
        # multi-agent notes for the case this was found from.
        yield "error", {
            "message": f"Stopped after {agent.max_steps} steps without a final answer "
            "(max_steps reached).",
        }

    yield "__stop__", {"stop_reason": stop_reason}


# -- helpers ---------------------------------------------------------------
def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def estimate_tokens_simple(text: str) -> int:
    return max(1, len(text or "") // 4)


# ---------------------------------------------------------------------------
# Main (for ``python app/server.py``, run from the repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
