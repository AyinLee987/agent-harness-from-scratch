"""Data contracts for leader/subagent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class TaskStatus(str, Enum):
    """Lifecycle states for one delegated task."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_TASK_STATUSES = {
    TaskStatus.SUCCEEDED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.TIMED_OUT,
}


@dataclass(frozen=True)
class AgentSpec:
    """Description and factory key for one available worker role."""

    name: str
    description: str
    can_delegate: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("AgentSpec.name must be non-empty.")
        if not self.description.strip():
            raise ValueError("AgentSpec.description must be non-empty.")


@dataclass(frozen=True)
class RunBudget:
    """Hard limits shared by all subagents beneath one leader run."""

    max_subagents: int = 8
    max_parallel_tasks: int = 3
    max_depth: int = 1
    max_repeated_task: int = 1
    subagent_timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.max_subagents < 1:
            raise ValueError("max_subagents must be at least 1.")
        if self.max_parallel_tasks < 1:
            raise ValueError("max_parallel_tasks must be at least 1.")
        if self.max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if self.max_repeated_task < 1:
            raise ValueError("max_repeated_task must be at least 1.")
        if self.subagent_timeout_seconds <= 0:
            raise ValueError("subagent_timeout_seconds must be positive.")


@dataclass(frozen=True)
class SubagentTask:
    """Serializable snapshot of a delegated task."""

    task_id: str
    root_run_id: str
    parent_run_id: str
    agent_name: str
    instruction: str
    status: TaskStatus
    depth: int
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class SubagentResult:
    """Normalized result returned from a Worker to its Leader."""

    task_id: str
    agent_name: str
    success: bool
    answer: str
    stop_reason: str
    error_type: Optional[str]
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    steps: int = 0
    tokens: int = 0

    def to_dict(self, include_trajectory: bool = False) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "task_id": self.task_id,
            "agent_name": self.agent_name,
            "success": self.success,
            "answer": self.answer,
            "stop_reason": self.stop_reason,
            "error_type": self.error_type,
            "steps": self.steps,
            "tokens": self.tokens,
        }
        if include_trajectory:
            data["trajectory"] = list(self.trajectory)
        return data


@dataclass(frozen=True)
class MultiAgentRunResult:
    """Leader result plus the final snapshot of every delegated task."""

    root_run_id: str
    answer: str
    success: bool
    steps: int
    tokens: int
    stop_reason: str
    trajectory: List[Dict[str, Any]]
    subagents: List[SubagentResult]

