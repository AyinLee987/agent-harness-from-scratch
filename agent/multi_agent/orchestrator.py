"""Thread-safe lifecycle manager for Leader-dispatched Worker agents."""

from __future__ import annotations

import hashlib
import logging
from contextlib import contextmanager
from contextvars import ContextVar
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

from ..agent import ReActAgent
from ..errors import FatalToolError, RecoverableToolError
from ..observability import bind_log_context, get_logger, log_event, run_log_file
from ..tools import BaseTool
from .models import (
    TERMINAL_TASK_STATUSES,
    MultiAgentRunResult,
    RunBudget,
    SubagentResult,
    SubagentTask,
    TaskStatus,
)
from .registry import AgentRegistry

logger = get_logger(__name__)


@dataclass
class _RootState:
    root_run_id: str
    task_ids: List[str] = field(default_factory=list)
    fingerprints: Counter[str] = field(default_factory=Counter)
    closed: bool = False


@dataclass
class _TaskRecord:
    task_id: str
    root_run_id: str
    parent_run_id: str
    agent_name: str
    instruction: str
    depth: int
    created_at: float
    status: TaskStatus = TaskStatus.PENDING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[SubagentResult] = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    timed_out: bool = False
    future: Optional[Future[None]] = None


class MultiAgentOrchestrator:
    """Runs Worker agents on behalf of a Leader through tool calls.

    Every registered factory must return a fresh :class:`ReActAgent`. A child
    fatal error only fails that child; it becomes structured data the Leader
    can reason about. Orchestrator invariant failures remain fatal to the root.
    """

    def __init__(
        self,
        registry: AgentRegistry,
        budget: Optional[RunBudget] = None,
    ) -> None:
        self.registry = registry
        self.budget = budget or RunBudget()
        self._executor = ThreadPoolExecutor(
            max_workers=self.budget.max_parallel_tasks,
            thread_name_prefix="subagent",
        )
        self._roots: Dict[str, _RootState] = {}
        self._tasks: Dict[str, _TaskRecord] = {}
        self._condition = threading.Condition(threading.RLock())
        self._active_root: ContextVar[Optional[str]] = ContextVar(
            f"multi_agent_root_{id(self)}", default=None
        )
        self._closed = False

    def leader_tools(self) -> List[BaseTool]:
        """Return spawn/status/wait/cancel tools for a Leader registry."""

        from .tools import create_leader_tools

        return create_leader_tools(self)

    def run_leader(
        self,
        leader: ReActAgent,
        task: str,
        *,
        resume_from: Optional[Dict[str, object]] = None,
    ) -> MultiAgentRunResult:
        """Run a Leader with an isolated root id and clean up orphan Workers."""

        with self.leader_scope() as root_run_id:
            leader_result = leader.run(task, resume_from=resume_from)

        subagents = self.results_for_run(root_run_id)
        return MultiAgentRunResult(
            root_run_id=root_run_id,
            answer=leader_result.answer,
            success=leader_result.success,
            steps=leader_result.steps,
            tokens=leader_result.tokens + sum(item.tokens for item in subagents),
            stop_reason=leader_result.stop_reason,
            trajectory=leader_result.trajectory,
            subagents=subagents,
            checkpoint=getattr(leader_result, "checkpoint", None),
            pending_job_ids=list(getattr(leader_result, "pending_job_ids", []) or []),
        )

    @contextmanager
    def leader_scope(self) -> Iterator[str]:
        """Activate delegation tools for one Leader execution context."""

        if self._closed:
            raise FatalToolError("Multi-agent orchestrator is closed.")
        if self._active_root.get() is not None:
            raise FatalToolError("Nested leader runs are not supported.")

        root_run_id = uuid.uuid4().hex[:12]
        with self._condition:
            self._roots[root_run_id] = _RootState(root_run_id=root_run_id)
        token = self._active_root.set(root_run_id)
        with bind_log_context(
            root_run_id=root_run_id, agent_name="leader"
        ), run_log_file(root_run_id, id_field="root_run_id"):
            log_event(logger, logging.INFO, "multi_agent.root.started")
            try:
                yield root_run_id
            finally:
                self._close_root(root_run_id)
                log_event(
                    logger,
                    logging.INFO,
                    "multi_agent.root.completed",
                    subagent_count=len(self.tasks_for_run(root_run_id)),
                )
                self._active_root.reset(token)

    def spawn_subagent(self, role: str, task: str) -> Dict[str, object]:
        """Start a Worker and return immediately with a task handle."""

        root_run_id = self._active_root_id()
        role = str(role or "").strip()
        instruction = str(task or "").strip()
        if not instruction:
            raise RecoverableToolError("Subagent task must be non-empty.")
        if role not in self.registry:
            raise RecoverableToolError(
                f"Unknown subagent role {role!r}. Available roles: "
                f"{', '.join(self.registry.names()) or '(none)'}."
            )

        fingerprint = self._fingerprint(role, instruction)
        with self._condition:
            root = self._roots.get(root_run_id)
            if root is None or root.closed:
                raise FatalToolError("Active multi-agent root is unavailable.")
            if len(root.task_ids) >= self.budget.max_subagents:
                raise RecoverableToolError(
                    f"Subagent budget exhausted ({self.budget.max_subagents})."
                )
            if 1 > self.budget.max_depth:
                raise RecoverableToolError(
                    f"Subagent depth limit reached ({self.budget.max_depth})."
                )
            if root.fingerprints[fingerprint] >= self.budget.max_repeated_task:
                raise RecoverableToolError(
                    "An equivalent subagent task was already dispatched; "
                    "reuse its task id or change the instruction."
                )

            task_id = uuid.uuid4().hex[:12]
            record = _TaskRecord(
                task_id=task_id,
                root_run_id=root_run_id,
                parent_run_id=root_run_id,
                agent_name=role,
                instruction=instruction,
                depth=1,
                created_at=time.time(),
            )
            self._tasks[task_id] = record
            root.task_ids.append(task_id)
            root.fingerprints[fingerprint] += 1
            record.future = self._executor.submit(self._execute_task, task_id)

        log_event(
            logger,
            logging.INFO,
            "subagent.dispatched",
            task_id=task_id,
            agent_name=role,
            instruction_chars=len(instruction),
        )

        return {
            "task_id": task_id,
            "agent_name": role,
            "status": TaskStatus.PENDING.value,
        }

    def get_subagent_status(self, task_id: str) -> Dict[str, object]:
        root_run_id = self._active_root_id()
        with self._condition:
            record = self._owned_task(root_run_id, task_id)
            data = self._snapshot(record).to_dict()
            if record.result is not None:
                data["result"] = record.result.to_dict()
            return data

    def wait_subagents(
        self,
        task_ids: Sequence[str],
        timeout_seconds: float = 60.0,
    ) -> List[Dict[str, object]]:
        root_run_id = self._active_root_id()
        if not isinstance(task_ids, (list, tuple)) or not task_ids:
            raise RecoverableToolError("task_ids must be a non-empty array.")
        if timeout_seconds <= 0:
            raise RecoverableToolError("timeout_seconds must be positive.")

        ids = [str(item) for item in task_ids]
        wait_started = time.monotonic()
        deadline = time.monotonic() + timeout_seconds
        with self._condition:
            for task_id in ids:
                self._owned_task(root_run_id, task_id)
            while True:
                records = [self._owned_task(root_run_id, item) for item in ids]
                if all(item.status in TERMINAL_TASK_STATUSES for item in records):
                    log_event(
                        logger,
                        logging.INFO,
                        "subagent.wait.completed",
                        task_ids=ids,
                        elapsed_ms=round((time.monotonic() - wait_started) * 1000, 2),
                    )
                    return [self._result_payload(item) for item in records]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    pending = [
                        item.task_id
                        for item in records
                        if item.status not in TERMINAL_TASK_STATUSES
                    ]
                    log_event(
                        logger,
                        logging.WARNING,
                        "subagent.wait.timed_out",
                        task_ids=ids,
                        pending_task_ids=pending,
                        timeout_seconds=timeout_seconds,
                    )
                    raise RecoverableToolError(
                        f"Timed out waiting for subagents: {', '.join(pending)}."
                    )
                self._condition.wait(timeout=remaining)

    def cancel_subagent(self, task_id: str) -> Dict[str, object]:
        root_run_id = self._active_root_id()
        with self._condition:
            record = self._owned_task(root_run_id, task_id)
            self._cancel_record(record, TaskStatus.CANCELLED)
            return self._snapshot(record).to_dict()

    def tasks_for_run(self, root_run_id: str) -> List[SubagentTask]:
        with self._condition:
            root = self._roots.get(root_run_id)
            if root is None:
                return []
            return [self._snapshot(self._tasks[item]) for item in root.task_ids]

    def results_for_run(self, root_run_id: str) -> List[SubagentResult]:
        with self._condition:
            root = self._roots.get(root_run_id)
            if root is None:
                return []
            return [self._normalized_result(self._tasks[item]) for item in root.task_ids]

    def result_payloads_for_run(
        self, root_run_id: str, *, include_trajectory: bool = False
    ) -> List[Dict[str, object]]:
        """Like :meth:`results_for_run`, but as JSON-ready dicts carrying each
        task's current ``status`` -- for HTTP/UI responses.

        ``SubagentResult`` (what :meth:`results_for_run` returns, and what a
        Leader sees from ``wait_subagents``/``get_subagent_status``) has no
        ``status`` field of its own; those two tools add it on top via
        :meth:`_result_payload`. The final summary a UI renders after a run
        (e.g. the ``/api/stream`` "done" event) needs the same treatment: a
        Worker the Leader dispatched but never checked on again (no
        ``wait_subagents``/``get_subagent_status`` call before finishing)
        would otherwise be reported with no status at all, which a client
        merging it onto its last-known "pending" card (from the
        ``spawn_subagent`` result) renders as stuck pending forever, even
        though the Worker actually succeeded, failed, or was cancelled when
        the root closed.
        """
        with self._condition:
            root = self._roots.get(root_run_id)
            if root is None:
                return []
            return [
                self._result_payload(self._tasks[item], include_trajectory=include_trajectory)
                for item in root.task_ids
            ]

    def close(self) -> None:
        """Cancel logical task handles and shut down Worker threads."""

        with self._condition:
            if self._closed:
                return
            self._closed = True
            for record in self._tasks.values():
                if record.status not in TERMINAL_TASK_STATUSES:
                    self._cancel_record(record, TaskStatus.CANCELLED)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "MultiAgentOrchestrator":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _execute_task(self, task_id: str) -> None:
        with self._condition:
            record = self._tasks[task_id]
            if record.status == TaskStatus.CANCELLED:
                return
            record.status = TaskStatus.RUNNING
            record.started_at = time.time()
            self._condition.notify_all()

        with bind_log_context(
            root_run_id=record.root_run_id,
            task_id=record.task_id,
            agent_name=record.agent_name,
        ):
            self._execute_task_in_context(record)

    def _execute_task_in_context(self, record: _TaskRecord) -> None:
        """Execute a Worker with correlation context propagated into its thread."""

        log_event(logger, logging.INFO, "subagent.started")

        timer = threading.Timer(
            self.budget.subagent_timeout_seconds,
            self._timeout_task,
            args=(record.task_id,),
        )
        timer.daemon = True
        timer.start()
        try:
            agent = self.registry.create(record.agent_name)
            outcome = agent.run(record.instruction, cancellation_event=record.cancel_event)
            result = SubagentResult(
                task_id=record.task_id,
                agent_name=record.agent_name,
                success=outcome.success,
                answer=outcome.answer,
                stop_reason=outcome.stop_reason,
                error_type=None if outcome.success else outcome.stop_reason.split(":", 1)[0],
                trajectory=outcome.trajectory,
                steps=outcome.steps,
                tokens=outcome.tokens,
            )
        except Exception as exc:
            log_event(
                logger,
                logging.ERROR,
                "subagent.failed",
                error_type=type(exc).__name__,
                exc_info=True,
            )
            result = SubagentResult(
                task_id=record.task_id,
                agent_name=record.agent_name,
                success=False,
                answer=f"Subagent failed: {exc}",
                stop_reason="subagent_internal_error",
                error_type=type(exc).__name__,
            )
        finally:
            timer.cancel()

        with self._condition:
            record = self._tasks[record.task_id]
            if record.status == TaskStatus.CANCELLED:
                self._condition.notify_all()
                return
            if record.timed_out:
                record.status = TaskStatus.TIMED_OUT
                record.result = SubagentResult(
                    task_id=record.task_id,
                    agent_name=record.agent_name,
                    success=False,
                    answer="Subagent exceeded its execution timeout.",
                    stop_reason="timed_out",
                    error_type="timeout",
                    trajectory=result.trajectory,
                    steps=result.steps,
                    tokens=result.tokens,
                )
            else:
                record.result = result
                record.status = (
                    TaskStatus.SUCCEEDED if result.success else TaskStatus.FAILED
                )
            record.finished_at = time.time()
            self._condition.notify_all()
            log_event(
                logger,
                logging.INFO if record.status == TaskStatus.SUCCEEDED else logging.WARNING,
                "subagent.completed",
                status=record.status.value,
                success=record.result.success if record.result else False,
                steps=record.result.steps if record.result else 0,
                tokens=record.result.tokens if record.result else 0,
                elapsed_ms=round(
                    ((record.finished_at or time.time()) - (record.started_at or record.created_at))
                    * 1000,
                    2,
                ),
            )

    def _timeout_task(self, task_id: str) -> None:
        with self._condition:
            record = self._tasks.get(task_id)
            if record is None or record.status in TERMINAL_TASK_STATUSES:
                return
            record.timed_out = True
            record.cancel_event.set()
            record.status = TaskStatus.TIMED_OUT
            record.finished_at = time.time()
            record.result = self._normalized_result(record)
            self._condition.notify_all()
            log_event(
                logger,
                logging.WARNING,
                "subagent.timed_out",
                task_id=record.task_id,
                agent_name=record.agent_name,
            )

    def _close_root(self, root_run_id: str) -> None:
        with self._condition:
            root = self._roots.get(root_run_id)
            if root is None:
                return
            root.closed = True
            for task_id in root.task_ids:
                record = self._tasks[task_id]
                if record.status not in TERMINAL_TASK_STATUSES:
                    self._cancel_record(record, TaskStatus.CANCELLED)
            self._condition.notify_all()

    def _cancel_record(self, record: _TaskRecord, status: TaskStatus) -> None:
        if record.status in TERMINAL_TASK_STATUSES:
            return
        record.cancel_event.set()
        if record.future is not None:
            record.future.cancel()
        record.status = status
        record.finished_at = time.time()
        record.result = self._normalized_result(record)
        self._condition.notify_all()
        log_event(
            logger,
            logging.INFO,
            "subagent.cancelled",
            task_id=record.task_id,
            agent_name=record.agent_name,
            status=status.value,
        )

    def _active_root_id(self) -> str:
        if self._closed:
            raise FatalToolError("Multi-agent orchestrator is closed.")
        root_run_id = self._active_root.get()
        if not root_run_id:
            raise FatalToolError(
                "Subagent tools must be called inside an active Leader run."
            )
        return str(root_run_id)

    def _owned_task(self, root_run_id: str, task_id: str) -> _TaskRecord:
        record = self._tasks.get(str(task_id))
        if record is None or record.root_run_id != root_run_id:
            raise RecoverableToolError(
                f"Unknown subagent task id {task_id!r} for this leader run."
            )
        return record

    @staticmethod
    def _fingerprint(role: str, instruction: str) -> str:
        normalized = " ".join(instruction.lower().split())
        return hashlib.sha256(f"{role}\0{normalized}".encode("utf-8")).hexdigest()

    @staticmethod
    def _snapshot(record: _TaskRecord) -> SubagentTask:
        return SubagentTask(
            task_id=record.task_id,
            root_run_id=record.root_run_id,
            parent_run_id=record.parent_run_id,
            agent_name=record.agent_name,
            instruction=record.instruction,
            status=record.status,
            depth=record.depth,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )

    @staticmethod
    def _normalized_result(record: _TaskRecord) -> SubagentResult:
        if record.result is not None:
            return record.result
        reason = record.status.value
        return SubagentResult(
            task_id=record.task_id,
            agent_name=record.agent_name,
            success=False,
            answer=f"Subagent task {reason}.",
            stop_reason=reason,
            error_type=reason,
        )

    def _result_payload(
        self, record: _TaskRecord, *, include_trajectory: bool = False
    ) -> Dict[str, object]:
        data: Dict[str, object] = self._normalized_result(record).to_dict(
            include_trajectory=include_trajectory
        )
        data["status"] = record.status.value
        return data
