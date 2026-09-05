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
import uuid
from contextlib import ExitStack, asynccontextmanager, contextmanager
from dataclasses import replace
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
    AgentGateway,
    AgentRegistry,
    AgentSpec,
    AgentResult,
    ContextProvider,
    DIRECT_SYSTEM_PROMPT,
    GatewayError,
    InMemoryCheckpointStore,
    CheckpointStore,
    JobBudget,
    JobRunner,
    LongRunningTool,
    RunCheckpoint,
    SQLiteCheckpointStore,
    SQLiteJobStore,
    create_job_tools,
    LexicalToolSelector,
    LLMQueryRouter,
    MultiAgentRunResult,
    QueueTimeout,
    RateLimitExceeded,
    Route,
    RunPlan,
    wants_escalation,
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
    RequestLoggingMiddleware,
    bind_log_context,
    configure_logging,
    get_logger,
    log_event,
    tool,
)
from agent.trigger.events import (
    ERROR,
    REFLECTION,
    RUN_COMPLETED,
    SUSPENDED,
    TEXT,
    THINK_COMPLETED,
    THINK_STARTED,
    TOOL_COMPLETED,
)
from agent.trigger.react_loop import SUSPENDED_STOP_REASON
from app.config import load_agent_config, retry_policy

configure_logging()
logger = get_logger("server")

# Leader/Worker run-tuning knobs (step/token budgets, delegation limits,
# retry/loop-detection behavior, session windowing) -- see config/agent.yaml
# and app/config.py. Loaded once at import time; edit the YAML and restart
# to pick up changes. AGENT_CONFIG_PATH points at a different file.
CONFIG = load_agent_config()

# Deadline/retry policy for every provider call this process makes -- see
# config/agent.yaml's `retry:` section and agent/retry.py for why the SDK
# defaults (600s, two blind retries) are not acceptable here.
RETRY_POLICY = retry_policy(CONFIG.retry)

# Process-wide admission control. `None` when config/agent.yaml sets
# `gateway.enabled: false`, which restores the pre-gateway behaviour of
# admitting every request immediately -- see _admission() below.
GATEWAY: Optional[AgentGateway] = (
    AgentGateway(
        rate_limit=CONFIG.gateway.rate_limit,
        rate_window_seconds=CONFIG.gateway.rate_window_seconds,
        max_concurrency=CONFIG.gateway.max_concurrency,
        queue_timeout=CONFIG.gateway.queue_timeout_seconds,
    )
    if CONFIG.gateway.enabled
    else None
)

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

# Long-running tool execution. Both default to the dependency-free
# in-memory/absent form for the same reason SESSION_STORE does -- importing
# this module (as a test calling route functions directly does) must never
# touch disk or start threads. lifespan() swaps in the durable versions.
JOB_RUNNER: Optional[JobRunner] = None
CHECKPOINT_STORE: CheckpointStore = InMemoryCheckpointStore()

def _auto_llm(spec):
    """The historical environment chain: DeepSeek > OpenAI > MockLLM."""

    if os.getenv("DEEPSEEK_API_KEY"):
        return DeepSeekLLM(
            model=spec.model, temperature=spec.temperature, retry_policy=RETRY_POLICY
        )
    if os.getenv("OPENAI_API_KEY") and os.getenv("USE_OPENAI"):
        from agent import OpenAILLM

        kwargs = {"temperature": spec.temperature, "retry_policy": RETRY_POLICY}
        if spec.model:
            kwargs["model"] = spec.model
        return OpenAILLM(**kwargs)
    return MockLLM()


def _llm_for(spec):
    """Build one model tier from its ``config/agent.yaml`` entry."""

    provider = (spec.provider or "auto").strip().lower()
    if provider == "auto":
        return _auto_llm(spec)
    if provider == "mock":
        return MockLLM()
    if provider == "deepseek":
        return DeepSeekLLM(
            model=spec.model, temperature=spec.temperature, retry_policy=RETRY_POLICY
        )
    if provider == "bailian":
        from agent import BailianLLM

        return BailianLLM(
            model=spec.model, temperature=spec.temperature, retry_policy=RETRY_POLICY
        )
    if provider in ("openai", "openai_compatible"):
        from agent import OpenAILLM

        api_key = os.getenv(spec.api_key_env) if spec.api_key_env else None
        if spec.api_key_env and not api_key:
            # Fail loud on setup (AGENTS.md §1.3): a tier pointed at an
            # unset key would otherwise only blow up on the first request
            # that happened to route to it.
            raise RuntimeError(
                f"config/agent.yaml points a model tier at {spec.api_key_env}, "
                f"which is not set."
            )
        kwargs = {
            "temperature": spec.temperature,
            "retry_policy": RETRY_POLICY,
            "base_url": spec.base_url,
            "api_key": api_key,
        }
        if spec.model:
            kwargs["model"] = spec.model
        return OpenAILLM(**kwargs)
    raise RuntimeError(
        f"Unknown model provider {spec.provider!r} in config/agent.yaml. "
        f"Use one of: auto, mock, deepseek, bailian, openai_compatible."
    )


def _build_llm():
    """The model the ReAct loop itself runs on (``models.react``)."""

    return _llm_for(CONFIG.models.react)


def _build_fast_llm():
    """The cheap tier (``models.fast``) for short, structured, verifiable calls.

    Falls back to :func:`_build_llm` while ``models.fast`` is left at
    ``provider: auto`` -- so a deployment that never configures a second
    tier behaves exactly as it did before, and so a test that patches
    ``_build_llm`` still controls every model in the process rather than
    having half of them silently escape to a real provider.
    """

    if (CONFIG.models.fast.provider or "auto").strip().lower() == "auto":
        return _build_llm()
    return _llm_for(CONFIG.models.fast)


@contextmanager
def _admission(trace_id: Optional[str] = None):
    """Hold a gateway slot for the duration of one agent run.

    A no-op when the gateway is disabled, so every call site can wrap
    itself unconditionally instead of branching. Must be entered from a
    worker thread -- ``AgentGateway.admit`` blocks while queueing.
    """

    if GATEWAY is None:
        yield None
        return
    with GATEWAY.admit(trace_id) as admission:
        yield admission


def _gateway_http_error(exc: GatewayError) -> HTTPException:
    """Map an admission rejection onto the status a client should act on.

    ``429`` says "you are asking too often, back off"; ``503`` says "the
    server is saturated, this request never started". Both carry
    ``Retry-After`` so a client has something better than a guess.
    """

    if isinstance(exc, RateLimitExceeded):
        return HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": str(max(1, int(exc.retry_after)))},
        )
    if isinstance(exc, QueueTimeout):
        return HTTPException(
            status_code=503,
            detail=str(exc),
            headers={"Retry-After": str(max(1, int(CONFIG.gateway.queue_timeout_seconds)))},
        )
    return HTTPException(status_code=503, detail=str(exc))


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
    decomposer = LLMQueryDecomposer(_build_fast_llm()) if _enabled("ENABLE_RAG_QUERY_DECOMPOSITION") else None
    # Opt-in: replace the default HeuristicReranker (lexical-overlap
    # heuristic) with one that asks the chat model itself to score each
    # candidate's relevance. Real cost: ~2.2s added per retrieval (one more
    # LLM call) -- measured on BEIR NFCorpus (see the companion evaluation
    # project's benchmarks/rag_recall_beir/RESULTS_llm_rerank.md) to beat
    # both plain RRF fusion and HeuristicReranker on every metric,
    # significantly on Recall@5/10 and nDCG@5/10 -- not on MRR. Off by
    # default because of that latency, not because it measures worse.
    reranker = LLMReranker(_build_fast_llm()) if _enabled("ENABLE_LLM_RERANKER") else None
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


def _tool_selector():
    """The Leader's tool selector, or ``None`` to offer every tool.

    Only the Leader gets one: a Worker's registry is already small and
    purpose-built (see ``build_researcher``/``build_analyst`` below), which
    is the same problem tool routing solves, solved by hand.
    """

    if not CONFIG.tool_router.enabled:
        return None
    return LexicalToolSelector(
        top_k=CONFIG.tool_router.top_k,
        min_tools=CONFIG.tool_router.min_tools,
        pinned=CONFIG.tool_router.pinned,
    )


def _copy_tools(*names: str) -> ToolRegistry:
    selected = [REGISTRY.get(name) for name in names]
    return ToolRegistry([item for item in selected if item is not None])


def _build_leader_runtime(
    max_steps: Optional[int] = None,
    extra_context_providers: Optional[Sequence[ContextProvider]] = None,
    attach_rag: bool = True,
):
    """Create a request-scoped Leader whose delegation ability is a tool set.

    ``max_steps=None`` (the default) lets the Leader and each Worker use
    their own independent step budget from ``config/agent.yaml``
    (``CONFIG.leader.max_steps`` / ``CONFIG.worker.max_steps``). Passing an
    explicit value (e.g. a client's ``POST /api/run`` request) overrides
    both with that one number, matching the pre-config behavior where a
    single ``max_steps`` applied to the whole run.

    ``attach_rag=False`` skips the *mandatory* evidence injection
    (``RAGContextProvider``) while still registering
    ``medical_evidence_search`` as a tool. That distinction is the whole
    point of the router's ``react`` route: a task that needs tools may or
    may not also need the corpus, and paying for a full retrieval before
    the model has decided is the cost this removes. It never removes the
    model's *ability* to retrieve.
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
    leader_registry = _register_job_tools(leader_registry)
    context_providers: List[ContextProvider] = []
    if RAG_PIPELINE and attach_rag:
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
        tool_selector=_tool_selector(),
        context_providers=context_providers or None,
        memory_manager=MEMORY_MANAGER,
        memory_namespace=MEMORY_NAMESPACE,
        memory_subject_id=MEMORY_SUBJECT_ID,
    )
    return orchestrator, leader


RAG_DOMAIN_HINT = os.getenv(
    "ROUTER_DOMAIN_HINT",
    "医疗健康：症状分诊、常见疾病、用药与就诊科室建议，证据来自权威医学资料库。",
)


def _plan_run(task: str) -> RunPlan:
    """Classify one request, or return the pre-router plan when disabled.

    The disabled path deliberately reports ``needs_retrieval=True``: with no
    router, ``RAGContextProvider`` ran on every single request, and turning
    the router off has to mean *exactly* the old behaviour, not a quietly
    different one.
    """

    if not CONFIG.router.enabled:
        return RunPlan(
            route=Route.REACT,
            task=task,
            original_task=task,
            needs_retrieval=True,
            reasoning="router disabled",
        )

    plan = LLMQueryRouter(
        _build_fast_llm(), domain_hint=RAG_DOMAIN_HINT if RAG_PIPELINE else ""
    ).route(task)
    if not CONFIG.router.use_rewrite:
        # Classification is the cheap, checkable half; rewriting is the half
        # that can silently drop something the user actually typed. They are
        # separately switchable so the first can ship without the second.
        plan = replace(plan, task=plan.original_task)
    return plan


def _run_direct(plan: RunPlan) -> Optional[MultiAgentRunResult]:
    """Answer a ``direct``-routed request with one tool-less model call.

    Returns ``None`` when the model replies with the escalation sentinel,
    which is the caller's signal to fall through to the full loop. That
    escape hatch is what makes the route safe to act on: the failure mode
    of a misroute is one wasted fast call, not a confidently wrong answer.
    """

    agent = ReActAgent(
        llm=_build_llm(),
        tools=ToolRegistry([]),
        system_prompt=DIRECT_SYSTEM_PROMPT,
        max_steps=CONFIG.router.direct_max_steps,
        max_tokens=CONFIG.leader.max_tokens,
        agent_name="direct",
    )
    result = agent.run(plan.task)
    if wants_escalation(result.answer):
        log_event(
            logger,
            logging.INFO,
            "router.route.escalated",
            reason="direct answer requested tools",
            task_chars=len(plan.task),
        )
        return None
    log_event(
        logger,
        logging.INFO,
        "router.route.direct_answered",
        steps=result.steps,
        tokens=result.tokens,
    )
    return MultiAgentRunResult(
        root_run_id="",
        answer=result.answer,
        success=result.success,
        steps=result.steps,
        tokens=result.tokens,
        stop_reason=result.stop_reason,
        trajectory=result.trajectory,
        subagents=[],
    )


def _start_session_store() -> SessionMemoryStore:
    """Swap the module-default in-memory session store for a durable one."""

    return SQLiteSessionStore(os.getenv("SESSION_DB_PATH", "data/sessions.sqlite"))


def _start_jobs():
    """Build the durable job runner and checkpoint store, if jobs are on."""

    if not CONFIG.jobs.enabled:
        return None, InMemoryCheckpointStore()
    runner = JobRunner(
        store=SQLiteJobStore(os.getenv("JOBS_DB_PATH", "data/jobs.sqlite")),
        budget=JobBudget(
            max_parallel_jobs=CONFIG.jobs.max_parallel_jobs,
            max_duration_seconds=CONFIG.jobs.max_duration_seconds,
            stall_timeout_seconds=CONFIG.jobs.stall_timeout_seconds,
            dedupe_ttl_seconds=CONFIG.jobs.dedupe_ttl_seconds,
        ),
    )
    checkpoints = SQLiteCheckpointStore(
        os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")
    )
    # Jobs left running by a previous process are visible but not adopted:
    # their worker threads died with that process, so reporting them as
    # still running would be a lie. Surfacing the count is the honest
    # middle ground -- an operator can see that work was lost, which an
    # in-memory-only design could not have told them at all.
    orphans = runner.store.list_unfinished()
    log_event(
        logger,
        logging.INFO,
        "jobs.started",
        long_running_tools=list(CONFIG.jobs.long_running),
        orphaned_jobs=len(orphans),
        suspended_runs=len(checkpoints.list_suspended()),
    )
    return runner, checkpoints


def _register_job_tools(registry: ToolRegistry) -> ToolRegistry:
    """Wrap the configured long-running tools and add the job control tools.

    Wrapping happens by re-registering under the same name, so the model's
    view of the tool -- its name and arguments -- is unchanged; only what
    it gets back differs. See :class:`~agent.jobs.LongRunningTool`.
    """

    if JOB_RUNNER is None:
        return registry

    wrapped: List[Any] = []
    for name in registry.names():
        item = registry.get(name)
        if item is None:
            continue
        wrapped.append(
            LongRunningTool(item, JOB_RUNNER)
            if name in CONFIG.jobs.long_running
            else item
        )
    rebuilt = ToolRegistry(wrapped)
    rebuilt.register_many(create_job_tools(
        JOB_RUNNER, default_wait_seconds=CONFIG.jobs.await_seconds
    ))
    return rebuilt


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Connect explicitly enabled MCP servers and close them on shutdown."""

    global MCP_MANAGER, RAG_REPOSITORY, RAG_PIPELINE, RAG_INGESTION, MEMORY_MANAGER, SESSION_STORE
    global JOB_RUNNER, CHECKPOINT_STORE
    manager = None
    try:
        await asyncio.to_thread(_start_rag)
        await asyncio.to_thread(_start_memory)
        SESSION_STORE = await asyncio.to_thread(_start_session_store)
        JOB_RUNNER, CHECKPOINT_STORE = await asyncio.to_thread(_start_jobs)
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
        if JOB_RUNNER is not None:
            # Jobs still running are cancelled rather than waited on: an
            # in-flight half-hour job would otherwise block shutdown for
            # half an hour. Their records stay in the store, so the next
            # process can see that they were interrupted.
            await asyncio.to_thread(JOB_RUNNER.close)
        JOB_RUNNER = None
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
    run_id: Optional[str] = Field(
        default=None,
        description="Set only when stop_reason is 'suspended_on_jobs'. The "
        "run is waiting on long-running work; POST /api/runs/{run_id}/resume "
        "to continue it once the jobs finish.",
    )
    pending_job_ids: list[str] = Field(
        default_factory=list,
        description="Jobs the suspended run is waiting on. Poll them with "
        "GET /api/jobs/{job_id}.",
    )

    @classmethod
    def from_result(
        cls,
        r: AgentResult,
        *,
        conversation_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> "RunResponse":
        return cls(
            answer=r.answer,
            success=r.success,
            steps=r.steps,
            tokens=r.tokens,
            stop_reason=r.stop_reason,
            trajectory=r.trajectory,
            conversation_id=conversation_id,
            run_id=run_id,
            pending_job_ids=list(getattr(r, "pending_job_ids", []) or []),
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
    """Health check plus the two saturation numbers worth watching.

    ``free_slots``/``queued`` are what tell a load balancer (or a human
    reading a dashboard) whether 503s are coming: a queue that is
    persistently non-empty means ``gateway.max_concurrency`` is below what
    this deployment actually needs.
    """

    llm = _build_llm()
    payload = {
        "status": "ok",
        "model": getattr(llm, "model", "mock"),
        "tools": len(REGISTRY),
    }
    if GATEWAY is not None:
        payload["gateway"] = {
            "max_concurrency": CONFIG.gateway.max_concurrency,
            "free_slots": GATEWAY.concurrency_guard.available,
            "queued": GATEWAY.queue.size,
            "rate_limit": CONFIG.gateway.rate_limit,
        }
    return payload


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
            llm=_build_fast_llm(),
            recent_window=CONFIG.session.recent_window,
            summarize_beyond=CONFIG.session.summarize_beyond,
        )
        extra_providers.append(session_provider)

    def execute():
        # Admission happens inside the worker thread because
        # AgentGateway.admit() blocks while queueing; doing it here rather
        # than in the coroutine keeps the event loop free.
        with _admission():
            plan = _plan_run(req.task)
            if plan.route is Route.DIRECT and CONFIG.router.act_on_direct:
                direct = _run_direct(plan)
                if direct is not None:
                    return direct
            orchestrator, leader = _build_leader_runtime(
                req.max_steps,
                extra_context_providers=extra_providers,
                attach_rag=plan.needs_retrieval,
            )
            with orchestrator:
                return orchestrator.run_leader(leader, plan.task)

    try:
        result = await asyncio.to_thread(execute)
    except GatewayError as exc:
        raise _gateway_http_error(exc) from exc

    if result.stop_reason == SUSPENDED_STOP_REASON:
        # Nothing is recorded on the conversation yet: the run has not
        # produced an answer, and persisting a half-finished one would
        # show up as the assistant's reply on the next reload.
        run_id = await asyncio.to_thread(
            _save_checkpoint, req.task, result, req.conversation_id
        )
        return RunResponse.from_result(
            result, conversation_id=req.conversation_id, run_id=run_id
        )

    if session_provider is not None:
        await asyncio.to_thread(session_provider.record_turn, req.task, result.answer)
    return RunResponse.from_result(result, conversation_id=req.conversation_id)


def _save_checkpoint(
    task: str, result, conversation_id: Optional[str]
) -> str:
    run_id = uuid.uuid4().hex[:12]
    CHECKPOINT_STORE.save(
        RunCheckpoint(
            run_id=run_id,
            task=task,
            checkpoint=result.checkpoint or {},
            pending_job_ids=list(result.pending_job_ids),
            conversation_id=conversation_id,
        )
    )
    log_event(
        logger,
        logging.INFO,
        "run.suspended.saved",
        run_id=run_id,
        pending_job_ids=list(result.pending_job_ids),
    )
    return run_id


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str):
    """Current state of one long-running job."""

    if JOB_RUNNER is None:
        raise HTTPException(503, "Long-running jobs are disabled (jobs.enabled).")
    job = await asyncio.to_thread(JOB_RUNNER.get, job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id {job_id!r}.")
    return job.to_dict()


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Stop a long-running job whose result is no longer needed."""

    if JOB_RUNNER is None:
        raise HTTPException(503, "Long-running jobs are disabled (jobs.enabled).")
    job = await asyncio.to_thread(JOB_RUNNER.cancel, job_id)
    if job is None:
        raise HTTPException(404, f"Unknown job_id {job_id!r}.")
    return job.to_dict(include_result=False)


@app.get("/api/runs/{run_id}")
async def suspended_run(run_id: str):
    """Whether a suspended run is ready to resume, and what it is waiting on."""

    stored = await asyncio.to_thread(CHECKPOINT_STORE.load, run_id)
    if stored is None:
        raise HTTPException(404, f"Unknown or already-resumed run_id {run_id!r}.")
    jobs = []
    if JOB_RUNNER is not None:
        for job_id in stored.pending_job_ids:
            job = await asyncio.to_thread(JOB_RUNNER.get, job_id)
            if job is not None:
                jobs.append(job.to_dict(include_result=False))
    return {
        "run_id": stored.run_id,
        "status": "suspended",
        "task": stored.task,
        "conversation_id": stored.conversation_id,
        "pending_jobs": jobs,
        "ready": bool(jobs) and all(job["status"] not in ("pending", "running") for job in jobs),
    }


@app.post("/api/runs/{run_id}/resume", response_model=RunResponse)
async def resume_run(run_id: str):
    """Continue a run that suspended waiting on long-running jobs.

    The checkpoint is deleted only once the run reaches a terminal state --
    a resume that suspends again saves a *new* checkpoint under the same
    id, so a job that finishes in two stages doesn't strand the run.
    """

    stored = await asyncio.to_thread(CHECKPOINT_STORE.load, run_id)
    if stored is None:
        raise HTTPException(404, f"Unknown or already-resumed run_id {run_id!r}.")

    def execute():
        with _admission():
            orchestrator, leader = _build_leader_runtime()
            with orchestrator:
                return orchestrator.run_leader(
                    leader, stored.task, resume_from=stored.checkpoint
                )

    try:
        result = await asyncio.to_thread(execute)
    except GatewayError as exc:
        raise _gateway_http_error(exc) from exc

    if result.stop_reason == SUSPENDED_STOP_REASON:
        stored.checkpoint = result.checkpoint or {}
        stored.pending_job_ids = list(result.pending_job_ids)
        await asyncio.to_thread(CHECKPOINT_STORE.save, stored)
        return RunResponse.from_result(
            result, conversation_id=stored.conversation_id, run_id=run_id
        )

    await asyncio.to_thread(CHECKPOINT_STORE.delete, run_id)
    if stored.conversation_id:
        provider = SessionContextProvider(
            SESSION_STORE,
            stored.conversation_id,
            llm=_build_fast_llm(),
            recent_window=CONFIG.session.recent_window,
            summarize_beyond=CONFIG.session.summarize_beyond,
        )
        await asyncio.to_thread(provider.record_turn, stored.task, result.answer)
    return RunResponse.from_result(result, conversation_id=stored.conversation_id)


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

    llm = _build_fast_llm()
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
            llm=_build_fast_llm(),
            recent_window=CONFIG.session.recent_window,
            summarize_beyond=CONFIG.session.summarize_beyond,
        )

    # Claim a gateway slot *before* the response starts, so a rejected
    # request gets a real 429/503 instead of an SSE stream whose first event
    # is an error -- and hold it until the generator finishes, since the run
    # occupies a concurrency slot for the whole stream, not just its setup.
    # ExitStack because the slot is acquired here and released inside the
    # generator's finally, on the other side of an async boundary.
    admission_stack = ExitStack()
    try:
        await asyncio.to_thread(admission_stack.enter_context, _admission())
    except GatewayError as exc:
        raise _gateway_http_error(exc) from exc

    plan = await asyncio.to_thread(_plan_run, task)
    take_direct = plan.route is Route.DIRECT and CONFIG.router.act_on_direct

    orchestrator, agent = _build_leader_runtime(
        extra_context_providers=[session_provider] if session_provider else None,
        attach_rag=plan.needs_retrieval,
    )
    leader_registry = agent.tools

    async def event_stream() -> AsyncGenerator[str, None]:
        # No ExecutionContext, no dispatcher, no think/act loop here any
        # more: the run is ReActLoop.aiter_run() and this endpoint only
        # forwards its events. Everything this used to set up by hand --
        # and get subtly wrong, see BUGS.md #22 -- now comes from the one
        # implementation POST /api/run also uses.
        stop_reason = "max_steps"
        steps = tokens = 0
        root_run_id = ""
        final_answer: Optional[str] = None
        suspended_run_id: Optional[str] = None
        result = None

        # Persist the question the moment the run starts, not after it
        # finishes — a disconnect (e.g. the client refreshing mid-stream) or
        # a run that never reaches a clean "answer" event (budget exhausted,
        # fatal tool error, cancelled) would otherwise leave nothing in
        # SESSION_STORE for this turn, so the conversation can look like it
        # vanished on the next reload even though it clearly happened.
        if session_provider is not None:
            await asyncio.to_thread(session_provider.record_user_message, task)

        try:
            if take_direct:
                # Buffered rather than streamed through, deliberately: a
                # direct answer that turns out to need tools escalates into
                # the full loop, and tokens already delivered to a browser
                # cannot be unsaid. Buffering makes the escalation entirely
                # invisible to the client -- no "discard what I just sent"
                # event to handle -- and costs nothing worth having, since
                # the direct route is by definition the short answer.
                buffered: List[tuple] = []
                stats = {"steps": 0, "tokens": 0}
                async for event, data in _stream_direct_answer(plan):
                    if event == "__stats__":
                        stats = data
                    else:
                        buffered.append((event, data))
                if buffered:
                    yield _sse("start", {
                        "task": task,
                        "tools": [],
                        "root_run_id": "",
                        "conversation_id": conversation_id,
                        "route": plan.route.value,
                    })
                    for event, data in buffered:
                        if event == "answer":
                            final_answer = data["text"]
                        yield _sse(event, data)
                    yield _sse("done", {
                        **stats,
                        "success": True,
                        "stop_reason": "finished",
                        "root_run_id": "",
                        "subagents": [],
                        "conversation_id": conversation_id,
                        "route": plan.route.value,
                    })
                    if session_provider is not None and final_answer is not None:
                        await asyncio.to_thread(
                            session_provider.record_assistant_message, final_answer
                        )
                    return

            with orchestrator.leader_scope() as root_run_id:
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

                async for name, data in _stream_leader_steps(task=task, agent=agent):
                    if name == "__stop__":
                        stop_reason = data["stop_reason"]
                        steps, tokens = data["steps"], data["tokens"]
                        result = data.get("result")
                        continue
                    if name == "answer":
                        final_answer = data["text"]
                    if name == "suspended":
                        # Announced here, but the resumable run_id is only
                        # known once the run has actually finished
                        # suspending and produced its checkpoint -- a second
                        # "suspended" event carrying it follows below.
                        continue
                    yield _sse(name, data)

                log_event(
                    logger,
                    logging.INFO,
                    "agent.stream.completed",
                    success=stop_reason == "finished",
                    stop_reason=stop_reason,
                    steps=steps,
                    tokens=tokens,
                    elapsed_ms=round((time.perf_counter() - stream_started) * 1000, 2),
                )

            if stop_reason == SUSPENDED_STOP_REASON and result is not None:
                suspended_run_id = await asyncio.to_thread(
                    _save_checkpoint, task, result, conversation_id
                )
                yield _sse("suspended", {
                    "run_id": suspended_run_id,
                    "pending_job_ids": list(result.pending_job_ids),
                    "resume_url": f"/api/runs/{suspended_run_id}/resume",
                })

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
                "steps": steps,
                "tokens": tokens + sum(item["tokens"] for item in subagents),
                "success": stop_reason == "finished",
                "stop_reason": stop_reason,
                "root_run_id": root_run_id,
                "run_id": suspended_run_id,
                "subagents": subagents,
                "conversation_id": conversation_id,
            })
        finally:
            await asyncio.to_thread(orchestrator.close)
            await asyncio.to_thread(admission_stack.close)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _stream_direct_answer(plan: RunPlan):
    """Stream one tool-less answer for a ``direct``-routed request.

    Yields the same ``(event, data)`` pairs as :func:`_stream_leader_steps`
    plus a final ``("__stats__", {...})``, and yields **nothing at all**
    when the model asks to escalate -- an empty result is the caller's
    signal to fall through to the full loop. Reusing
    :func:`_stream_leader_steps` with an empty registry rather than writing
    a second think loop keeps the reflection/budget/logging behaviour
    identical on both paths.
    """

    direct_agent = ReActAgent(
        llm=_build_llm(),
        tools=ToolRegistry([]),
        system_prompt=DIRECT_SYSTEM_PROMPT,
        max_steps=CONFIG.router.direct_max_steps,
        max_tokens=CONFIG.leader.max_tokens,
        agent_name="direct",
    )

    collected: List[tuple] = []
    answer: Optional[str] = None
    stats = {"steps": 0, "tokens": 0}
    async for event, data in _stream_leader_steps(task=plan.task, agent=direct_agent):
        if event == "__stop__":
            stats = {"steps": data["steps"], "tokens": data["tokens"]}
            continue
        if event == "answer":
            answer = data["text"]
        collected.append((event, data))

    if wants_escalation(answer):
        log_event(
            logger,
            logging.INFO,
            "router.route.escalated",
            reason="direct answer requested tools",
            task_chars=len(plan.task),
        )
        return

    log_event(
        logger,
        logging.INFO,
        "router.route.direct_answered",
        steps=stats["steps"],
        tokens=stats["tokens"],
    )

    for payload in collected:
        yield payload
    yield "__stats__", stats


async def _stream_leader_steps(*, task, agent):
    """Translate one run's events into the SSE pairs the playground reads.

    This used to be a hand-written second copy of the ReAct loop -- the only
    way to observe a run *while* it happened. The copies drifted (BUGS.md
    #22): tool dispatch blocked the event loop, ``SuspendRun`` escaped
    uncaught with no checkpoint saved, and the reported step count was
    always zero because this loop counted with a local variable and never
    created a ``Step``.

    All of that is gone because none of it lives here any more. The run is
    ``ReActLoop.aiter_run()``; this function only renames its events. Both
    endpoints now execute the same state machine and differ in transport,
    which is the only thing they were ever meant to differ in.

    Yields ``(event, data)`` pairs, ending with ``("__stop__", {...})``
    carrying the finished :class:`AgentResult` under ``result``.
    """

    reported_error = False

    async for event in agent.aiter_run(task):
        kind, data = event.kind, event.data

        if kind == THINK_STARTED:
            yield "think_start", {"step": data["step"]}

        elif kind == TEXT:
            yield "text", data

        elif kind == THINK_COMPLETED:
            calls = data.get("tool_calls") or []
            for call in calls:
                yield "tool_call", {"step": data["step"], "tool": call}
            if calls:
                yield "act_start", {
                    "step": data["step"],
                    "tools": [call["name"] for call in calls],
                }

        elif kind == REFLECTION:
            yield "reflection", data

        elif kind == TOOL_COMPLETED:
            yield "tool_result", {
                "step": data["step"],
                "tool": data["name"],
                "result": data["observation"],
            }

        elif kind == SUSPENDED:
            # New on this path. Previously SuspendRun propagated out of the
            # generator, so the stream died mid-run with no run id to resume
            # and the jobs it was waiting on were stranded.
            yield "suspended", data

        elif kind == ERROR:
            reported_error = True
            yield "error", data

        elif kind == RUN_COMPLETED:
            if data["success"]:
                yield "answer", {"text": data["answer"]}
            elif data["stop_reason"] != SUSPENDED_STOP_REASON and not reported_error:
                # Budget, loop detection, cancellation and "ran out of steps
                # without answering" all end the run without any node having
                # emitted an error. Saying so explicitly matters: a run that
                # genuinely exhausted its budget used to look exactly like a
                # dropped connection from the frontend's side.
                yield "error", {"message": _stop_reason_message(data, agent)}
            yield "__stop__", data


def _stop_reason_message(data: dict, agent) -> str:
    """A sentence a user can act on, for a run that stopped without answering.

    The stop reasons themselves now come from ``ReActLoop`` rather than
    being invented here, so this only has to render them -- one vocabulary
    for both endpoints instead of the streaming path's own bare
    ``"max_steps"`` versus ``/api/run``'s ``"budget: max_steps (N) reached"``.
    """

    stop_reason = str(data.get("stop_reason", ""))
    detail = stop_reason.split(":", 1)[-1].strip()

    if "max_steps" in stop_reason:
        return (
            f"Stopped after {data.get('steps', agent.max_steps)} steps without "
            f"a final answer (max_steps reached)."
        )
    if stop_reason.startswith("budget"):
        return f"Budget exceeded: {detail}"
    if stop_reason.startswith("loop_detected"):
        return f"Stopped making progress: {detail}."
    if stop_reason == "cancelled":
        return "Run cancelled."
    if stop_reason == "llm_unavailable":
        return "The model provider was unreachable after retries."
    if stop_reason == "no_answer":
        return "The run ended without producing a final answer."
    return f"Run stopped: {stop_reason or 'unknown reason'}."


# -- helpers ---------------------------------------------------------------
def _sse(event: str, data: dict) -> str:
    """Format a dict as an SSE message."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Main (for ``python app/server.py``, run from the repo root)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
