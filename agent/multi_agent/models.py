"""Data contracts for leader/subagent orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..trigger.dispatch import is_failure_observation


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
            "tool_calls": self.tool_call_summary(),
        }
        if include_trajectory:
            data["trajectory"] = list(self.trajectory)
        return data

    def tool_call_summary(self) -> List[Dict[str, Any]]:
        """One {tool, ok} entry per step that made a tool call.

        Cheap enough to always include -- unlike the full trajectory, whose
        observations can be large (a single fetch result can run to
        thousands of characters) -- so a Leader deciding what to do next,
        and a UI rendering a Worker's card, can both at least tell *that* a
        Worker chained multiple tool calls (e.g. several fetches, some
        failing) without paying for every call's full output text. Set
        ``include_trajectory=True`` on :meth:`to_dict` for the full detail.

        ``step["error"]`` is *not* the right signal here -- ToolDispatcher
        (agent/trigger/dispatch.py) only ever sets it for a FatalToolError,
        which already aborts the run. A RecoverableToolError, an unknown
        tool, or malformed arguments all come back as a normal observation
        string the model is meant to read and react to -- dispatch.py's own
        convention for all of those is to prefix the text with "ERROR", so
        that's what "ok" checks here too. One real limitation: a tool whose
        genuinely successful output happens to start with the literal word
        "ERROR" would be misclassified -- none of this kit's tools do, but a
        custom tool could.
        """
        summary: List[Dict[str, Any]] = []
        for step in self.trajectory:
            action = step.get("action")
            if not action:
                continue
            ok = step.get("error") is None and not is_failure_observation(step.get("observation"))
            summary.append({"tool": action.get("name"), "ok": ok})
        return summary


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

