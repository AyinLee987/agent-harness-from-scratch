"""Loads ``config/agent.yaml`` — the tunable knobs for how the Leader and its
Workers run (step/token budgets, delegation limits, retry/loop-detection
behavior, session history windowing).

These used to be scattered hardcoded defaults inside ``app/server.py``
(``max_steps=10``, ``RunBudget(max_subagents=6, ...)``, etc.). This module
gives them one file to live in instead. Secrets and deployment-shaped
feature toggles (API keys, ``ENABLE_*``, DB paths) stay in ``.env`` — this is
only for numbers that shape one run's behavior, not what's enabled or how to
reach a service.

The YAML file is optional in every sense: the whole file, any section, or
any individual key can be missing, and the dataclass field default below
(the shipped default, mirrored in ``config/agent.yaml``'s comments) is used
instead. Nothing here is hot-reloaded — restart the process to pick up edits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

from agent.retry import RetryPolicy

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "agent.yaml"

T = TypeVar("T")


@dataclass(frozen=True)
class LeaderConfig:
    max_steps: int = 100
    max_tokens: int = 100_000


@dataclass(frozen=True)
class WorkerConfig:
    max_steps: int = 100


@dataclass(frozen=True)
class RunBudgetConfig:
    max_subagents: int = 6
    max_parallel_tasks: int = 3
    max_depth: int = 1
    max_repeated_task: int = 1
    subagent_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class ReActLoopConfig:
    max_tool_retries: int = 1
    loop_same_call_limit: int = 3
    compress_at_fraction: float = 0.6
    source_failure_hint_threshold: int = 2


@dataclass(frozen=True)
class SessionConfig:
    recent_window: int = 12
    summarize_beyond: int = 24


@dataclass(frozen=True)
class ModelConfig:
    """One model tier: which endpoint, which model, whose key.

    ``provider: auto`` (the default for both tiers) means "keep the
    historical environment-variable chain" -- DeepSeek if ``DEEPSEEK_API_KEY``
    is set, else OpenAI if ``OPENAI_API_KEY`` and ``USE_OPENAI`` are, else
    ``MockLLM``. Nothing about model selection changes until a tier is
    explicitly configured.

    ``provider: openai_compatible`` is the general case: DeepSeek, Bailian,
    Ollama, vLLM and OpenAI itself all speak the same protocol, so a
    ``base_url`` + ``api_key_env`` + ``model`` triple reaches any of them
    without needing a class per vendor.
    """

    provider: str = "auto"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    temperature: float = 0.0


@dataclass(frozen=True)
class ModelsConfig:
    """The two tiers a run draws from.

    Splitting them is a cost decision with a quality floor: the ReAct loop
    -- where a wrong token compounds across every later step -- keeps the
    strong model, while the short, structured, easily-verified calls
    (intent routing, query rewriting, conversation titles, history
    summarization) can run on a cheaper one.
    """

    react: ModelConfig = field(default_factory=ModelConfig)
    fast: ModelConfig = field(default_factory=ModelConfig)


@dataclass(frozen=True)
class RouterConfig:
    """Intent classification in front of the loop -- see
    :mod:`agent.trigger.router`. Off by default: it costs one extra
    fast-tier call per request, which only pays for itself once enough
    traffic is chit-chat that skipping the full run is a real saving.

    ``act_on_direct: false`` keeps the classification and its logging while
    still routing everything through the loop, so routing accuracy can be
    measured on real traffic *before* it is allowed to change behaviour.
    """

    enabled: bool = False
    act_on_direct: bool = True
    use_rewrite: bool = False
    direct_max_steps: int = 1


@dataclass(frozen=True)
class ToolRouterConfig:
    """Retrieval over tool descriptions -- see :mod:`agent.trigger.tool_router`.

    ``min_tools`` doubles as the on switch in practice: a registry that
    never exceeds it never goes through selection at all, so leaving this
    section alone changes nothing for a deployment with a handful of tools.
    """

    enabled: bool = False
    top_k: int = 8
    min_tools: int = 12
    #: Always offered regardless of score -- control-plane tools no task
    #: text will ever lexically match, but whose absence changes what the
    #: agent is structurally able to do.
    pinned: List[str] = field(
        default_factory=lambda: [
            "spawn_subagent",
            "get_subagent_status",
            "wait_subagents",
            "cancel_subagent",
        ]
    )


@dataclass(frozen=True)
class JobsConfig:
    """Long-running tool execution -- see :mod:`agent.jobs`.

    ``long_running`` names the tools whose calls should return a handle
    instead of blocking. It is a list rather than a per-tool flag because
    whether a tool is "long" depends on the deployment, not on the tool:
    the same ``fetch`` is milliseconds against a local mock and minutes
    against a slow site behind a rate limiter.
    """

    enabled: bool = False
    long_running: List[str] = field(default_factory=list)
    max_parallel_jobs: int = 4
    max_duration_seconds: float = 3600.0
    stall_timeout_seconds: float = 300.0
    dedupe_ttl_seconds: float = 900.0
    #: How long ``await_jobs`` waits inline before suspending the run.
    #: Short on purpose: holding a request open is the thing this whole
    #: mechanism exists to avoid, so waiting is a convenience for jobs that
    #: happen to be quick, not the primary path.
    await_seconds: float = 30.0


@dataclass(frozen=True)
class RetryConfig:
    """Mirror of :class:`agent.retry.RetryPolicy`, minus its validation.

    Kept as its own dataclass rather than reusing ``RetryPolicy`` directly
    so a bad YAML value fails at :func:`app.config.retry_policy` -- one
    obvious place, with the field name in the message -- instead of inside
    the generic ``_build`` helper shared by every other section.
    """

    timeout_seconds: float = 60.0
    max_attempts: int = 3
    initial_backoff: float = 0.5
    backoff_multiplier: float = 2.0
    max_backoff: float = 8.0
    jitter: float = 0.25
    total_deadline_seconds: Optional[float] = 180.0


@dataclass(frozen=True)
class GatewayConfig:
    """Admission control for the process as a whole -- see
    :class:`agent.trigger.AgentGateway`. ``enabled: false`` restores the
    pre-gateway behaviour of admitting every request immediately."""

    enabled: bool = True
    rate_limit: int = 100
    rate_window_seconds: float = 1.0
    max_concurrency: int = 8
    queue_timeout_seconds: float = 30.0


@dataclass(frozen=True)
class AgentConfig:
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    run_budget: RunBudgetConfig = field(default_factory=RunBudgetConfig)
    react_loop: ReActLoopConfig = field(default_factory=ReActLoopConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    tool_router: ToolRouterConfig = field(default_factory=ToolRouterConfig)
    jobs: JobsConfig = field(default_factory=JobsConfig)


def _build(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
    """Build a dataclass from a dict, ignoring unknown keys and leaving
    anything missing at its field default — a partial or stale YAML file
    (an old key removed upstream, a typo'd new one) should still load
    instead of crashing the server on startup."""

    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


def load_agent_config(path: Optional[os.PathLike] = None) -> AgentConfig:
    """Load ``config/agent.yaml`` (or ``AGENT_CONFIG_PATH``, or an explicit
    ``path``). A missing or empty file is not an error — it just means every
    field above keeps its shipped default."""

    config_path = Path(path or os.getenv("AGENT_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    raw: Dict[str, Any] = {}
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded:
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"{config_path} must contain a YAML mapping at the top level, "
                    f"got {type(loaded).__name__}."
                )
            raw = loaded

    return AgentConfig(
        leader=_build(LeaderConfig, raw.get("leader")),
        worker=_build(WorkerConfig, raw.get("worker")),
        run_budget=_build(RunBudgetConfig, raw.get("run_budget")),
        react_loop=_build(ReActLoopConfig, raw.get("react_loop")),
        session=_build(SessionConfig, raw.get("session")),
        retry=_build(RetryConfig, raw.get("retry")),
        gateway=_build(GatewayConfig, raw.get("gateway")),
        models=_build_models(raw.get("models")),
        router=_build(RouterConfig, raw.get("router")),
        tool_router=_build(ToolRouterConfig, raw.get("tool_router")),
        jobs=_build(JobsConfig, raw.get("jobs")),
    )


def _build_models(data: Optional[Dict[str, Any]]) -> ModelsConfig:
    """``models:`` is the one nested section, so it needs its own builder."""

    data = data or {}
    return ModelsConfig(
        react=_build(ModelConfig, data.get("react")),
        fast=_build(ModelConfig, data.get("fast")),
    )


def retry_policy(config: RetryConfig) -> RetryPolicy:
    """Turn the YAML-shaped :class:`RetryConfig` into the runtime policy."""

    try:
        return RetryPolicy(
            timeout_seconds=float(config.timeout_seconds),
            max_attempts=int(config.max_attempts),
            initial_backoff=float(config.initial_backoff),
            backoff_multiplier=float(config.backoff_multiplier),
            max_backoff=float(config.max_backoff),
            jitter=float(config.jitter),
            total_deadline_seconds=(
                None
                if config.total_deadline_seconds is None
                else float(config.total_deadline_seconds)
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid `retry:` section in agent.yaml: {exc}") from exc
