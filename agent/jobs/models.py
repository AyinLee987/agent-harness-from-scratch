"""Data contracts for long-running tool execution.

A tool that takes half an hour cannot be a normal tool call. The dispatcher
(`agent/trigger/dispatch.py`) is synchronous by design, so a blocking tool
holds the whole ReAct loop -- and, above it, the HTTP request -- for its
entire duration, with no way to report progress, survive a restart, or be
cancelled between the loop's own step boundaries.

The contract here breaks that: a long-running tool returns a **handle**
immediately, and the work continues on its own. The guiding rule, and the
one sentence worth remembering from this package:

    A tool's execution time must not appear on the agent's call stack.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..errors import ControlSignal


class JobStatus(str, Enum):
    """Lifecycle states for one long-running tool execution."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


TERMINAL_JOB_STATUSES = {
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.TIMED_OUT,
}


@dataclass
class Job:
    """One long-running tool execution.

    Attributes:
        job_id: Short correlation id, generated like every other entity id
            in this repo (``uuid.uuid4().hex[:12]``).
        tool_name: The wrapped tool this job runs.
        fingerprint: Hash of ``(tool_name, canonical arguments)``. The
            idempotency key: re-submitting the same work inside the TTL
            returns the existing job instead of starting a second
            half-hour run. Same mechanism as
            :meth:`~agent.multi_agent.MultiAgentOrchestrator._fingerprint`,
            which solves the same problem for delegation.
        run_id: The agent run that submitted it, so a resumed run can find
            its own jobs again.
        heartbeat_at: Last progress report. Distinct from ``started_at``
            because "running for 40 minutes" and "hasn't moved in 40
            minutes" are different conditions, and only the second is a
            reason to give up on a job that was expected to be slow.
        result: The observation text, once succeeded.
        error: Why it failed, if it did.
    """

    job_id: str
    tool_name: str
    fingerprint: str
    run_id: str = ""
    arguments: Dict[str, Any] = field(default_factory=dict)
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    heartbeat_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: str = ""
    result: Optional[str] = None
    error: Optional[str] = None

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_JOB_STATUSES

    def elapsed_seconds(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - (self.started_at or self.created_at))

    def to_dict(self, *, include_result: bool = True) -> Dict[str, Any]:
        """JSON-ready form. This is also what the model sees, so it stays
        small: a half-hour job's output can be large, and a status poll
        should not drag it through the context window every time."""

        data: Dict[str, Any] = {
            "job_id": self.job_id,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "elapsed_seconds": round(self.elapsed_seconds(), 1),
            "progress": self.progress,
        }
        if self.error:
            data["error"] = self.error
        if include_result and self.result is not None:
            data["result"] = self.result
        return data

    def to_record(self) -> Dict[str, Any]:
        """Full form for the store -- everything needed to rebuild the job."""

        return {
            "job_id": self.job_id,
            "tool_name": self.tool_name,
            "fingerprint": self.fingerprint,
            "run_id": self.run_id,
            "arguments": self.arguments,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "heartbeat_at": self.heartbeat_at,
            "finished_at": self.finished_at,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_record(cls, record: Dict[str, Any]) -> "Job":
        """Rebuild from a stored record, tolerating unknown/missing keys.

        Version-tolerant on purpose (AGENTS.md §10.3): a job written by an
        older build must still load rather than crash a resume.
        """

        return cls(
            job_id=str(record["job_id"]),
            tool_name=str(record.get("tool_name", "")),
            fingerprint=str(record.get("fingerprint", "")),
            run_id=str(record.get("run_id", "")),
            arguments=dict(record.get("arguments") or {}),
            status=JobStatus(str(record.get("status", JobStatus.PENDING.value))),
            created_at=float(record.get("created_at") or time.time()),
            started_at=record.get("started_at"),
            heartbeat_at=record.get("heartbeat_at"),
            finished_at=record.get("finished_at"),
            progress=str(record.get("progress") or ""),
            result=record.get("result"),
            error=record.get("error"),
        )


class SuspendRun(ControlSignal):
    """Control-flow signal: stop this run, it is waiting on jobs.

    A :class:`~agent.errors.ControlSignal`, deliberately not a
    :class:`~agent.errors.ToolCallError`. The two-tier tool taxonomy
    (recoverable / fatal) classifies *failures*, and this is not one --
    nothing went wrong, the run simply has nothing to do until a job
    finishes. Making it a third failure tier would blur a distinction
    AGENTS.md §6 is explicit about keeping sharp.

    Raised by the ``await_jobs`` tool and caught in
    :meth:`~agent.trigger.react_loop.ReActLoop._act_node`.
    """

    def __init__(self, job_ids: list) -> None:
        self.job_ids = list(job_ids)
        super().__init__(f"Run suspended waiting on jobs: {', '.join(self.job_ids)}")
