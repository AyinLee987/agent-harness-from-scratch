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
from typing import Any, Dict, Optional, Type, TypeVar

import yaml

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
class AgentConfig:
    leader: LeaderConfig = field(default_factory=LeaderConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    run_budget: RunBudgetConfig = field(default_factory=RunBudgetConfig)
    react_loop: ReActLoopConfig = field(default_factory=ReActLoopConfig)
    session: SessionConfig = field(default_factory=SessionConfig)


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
    )
