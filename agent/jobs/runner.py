"""Executes long-running tools off the agent's call stack.

Guardrails here follow AGENTS.md §8 -- every unbounded resource this
introduces gets an explicit, testable limit in the same place:

* ``max_parallel_jobs`` caps how many run at once;
* ``max_duration_seconds`` is a hard ceiling on any one job;
* ``stall_timeout_seconds`` gives up on a job that has stopped reporting
  progress, which is a *different* condition from one that is simply slow
  -- a legitimately 40-minute job must not be killed for taking 40 minutes;
* ``dedupe_ttl_seconds`` makes re-submission idempotent, so a retrying
  model cannot start the same half-hour of work twice.

Cancellation is cooperative and reaches the tool itself through
:class:`JobContext`. That is the part the multi-agent orchestrator gets
wrong today: it sets a ``threading.Event`` the ReAct loop only checks
between steps, so a task blocked inside a tool cannot actually be stopped.
A tool that accepts a ``job_context`` argument can check
:attr:`JobContext.cancelled` at its own checkpoints and stop for real.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..errors import ControlSignal, RecoverableToolError
from ..observability import get_logger, log_event
from ..tools import BaseTool
from .models import TERMINAL_JOB_STATUSES, Job, JobStatus
from .store import InMemoryJobStore, JobStore

logger = get_logger(__name__)


@dataclass(frozen=True)
class JobBudget:
    """Limits shared by every job one runner executes."""

    max_parallel_jobs: int = 4
    max_duration_seconds: float = 3600.0
    stall_timeout_seconds: float = 300.0
    dedupe_ttl_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.max_parallel_jobs < 1:
            raise ValueError("max_parallel_jobs must be at least 1.")
        if self.max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds must be positive.")
        if self.stall_timeout_seconds <= 0:
            raise ValueError("stall_timeout_seconds must be positive.")


class JobContext:
    """What a long-running tool gets so it can report in and be stopped.

    A tool opts in simply by accepting a ``job_context`` keyword argument;
    :class:`JobRunner` inspects the signature and only passes one to tools
    that ask. A tool that ignores it still works -- it just cannot be
    cancelled mid-call, and is governed by ``max_duration_seconds`` alone
    rather than by stall detection.
    """

    def __init__(self, job_id: str, cancel_event: threading.Event, on_heartbeat) -> None:
        self.job_id = job_id
        self._cancel_event = cancel_event
        self._on_heartbeat = on_heartbeat

    @property
    def cancelled(self) -> bool:
        """Check this at your own checkpoints and return/raise promptly."""

        return self._cancel_event.is_set()

    def heartbeat(self, progress: str = "") -> None:
        """Report that work is still happening, optionally with a note."""

        self._on_heartbeat(progress)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled(self.job_id)


class JobCancelled(ControlSignal):
    """Raised inside a tool by :meth:`JobContext.raise_if_cancelled`.

    A :class:`~agent.errors.ControlSignal` rather than a plain exception so
    :class:`~agent.tools.FunctionTool` lets it through untouched. As an
    ordinary exception it was being classified as a ``FatalToolError`` on
    the way out, which both hid it from the runner's own ``except
    JobCancelled`` and logged a deliberate cancellation as ``job.failed``.
    """

    def __init__(self, job_id: str) -> None:
        super().__init__(f"Job {job_id} was cancelled.")


def fingerprint(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Idempotency key for ``(tool, arguments)``.

    Key order is normalized so ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}``
    hash identically. Values are **not** normalized.

    This deliberately no longer matches ``_detect_loop``'s canonicalization
    in ``agent/trigger/react_loop.py``, which it was originally copied from.
    That function lowercases the whole argument blob, and for a *heuristic*
    that is fine: over-matching there costs at worst one early stop. Here
    the same lowercasing was a correctness bug, because this key decides
    whether to hand back an existing job's result: ``text="Alpha"`` and
    ``text="alpha"`` collided, so the second submission returned the first
    one's output. Case-sensitive URL paths, query strings and file contents
    were all affected. A heuristic's canonicalization does not transfer to
    an idempotency key -- see BUGS.md #11.
    """

    try:
        canon = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    except (TypeError, ValueError):
        canon = str(arguments)
    return hashlib.sha256(f"{tool_name}\0{canon}".encode("utf-8")).hexdigest()


class JobRunner:
    """Owns the threads, the store, and the lifecycle of long-running tools.

    Args:
        store: Where jobs are recorded. Defaults to in-memory; production
            should pass a :class:`~agent.jobs.SQLiteJobStore` so a restart
            does not lose track of work already in flight.
        budget: Limits shared by every job.
    """

    def __init__(
        self,
        store: Optional[JobStore] = None,
        budget: Optional[JobBudget] = None,
    ) -> None:
        self.store = store or InMemoryJobStore()
        self.budget = budget or JobBudget()
        self._executor = ThreadPoolExecutor(
            max_workers=self.budget.max_parallel_jobs, thread_name_prefix="job"
        )
        self._condition = threading.Condition(threading.RLock())
        self._cancel_events: Dict[str, threading.Event] = {}
        self._futures: Dict[str, Future] = {}
        self._closed = False

    # -- submission ---------------------------------------------------------
    def submit(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        *,
        run_id: str = "",
    ) -> Job:
        """Start ``tool`` in the background and return its handle immediately.

        Re-submitting work whose fingerprint matches a job started inside
        ``dedupe_ttl_seconds`` returns *that* job instead of starting a
        second one. Failed jobs are excluded from reuse: a retry after a
        failure is the one case where the model genuinely does want the
        work done again.
        """

        if self._closed:
            raise RecoverableToolError("Job runner is closed.")

        key = fingerprint(tool.name, arguments)
        existing = self.store.find_reusable(key, ttl_seconds=self.budget.dedupe_ttl_seconds)
        if existing is not None:
            log_event(
                logger,
                logging.INFO,
                "job.submit.deduplicated",
                job_id=existing.job_id,
                tool_name=tool.name,
                status=existing.status.value,
            )
            return existing

        job = Job(
            job_id=uuid.uuid4().hex[:12],
            tool_name=tool.name,
            fingerprint=key,
            run_id=run_id,
            arguments=dict(arguments),
        )
        self.store.put(job)
        with self._condition:
            self._cancel_events[job.job_id] = threading.Event()
            self._futures[job.job_id] = self._executor.submit(
                self._execute, job.job_id, tool, dict(arguments)
            )
        log_event(
            logger,
            logging.INFO,
            "job.submit.accepted",
            job_id=job.job_id,
            tool_name=tool.name,
            argument_keys=sorted(arguments.keys()),
        )
        return job

    # -- inspection ---------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        return self.store.get(job_id)

    def await_jobs(
        self, job_ids: Sequence[str], timeout_seconds: float
    ) -> List[Job]:
        """Block until every job is terminal, or ``timeout_seconds`` passes.

        Returns whatever state the jobs are in when it gives up -- the
        caller decides whether an unfinished job means "keep waiting",
        "suspend the run", or "report what we have". Deliberately not an
        error: on a long-running job path, not being done yet is the
        expected case, not a failure.
        """

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        deadline = time.monotonic() + timeout_seconds
        ids = [str(item) for item in job_ids]
        while True:
            jobs = [self.store.get(item) for item in ids]
            found = [job for job in jobs if job is not None]
            if all(job.status in TERMINAL_JOB_STATUSES for job in found) and len(
                found
            ) == len(ids):
                return found
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return found
            with self._condition:
                self._condition.wait(timeout=min(remaining, 0.5))

    def cancel(self, job_id: str) -> Optional[Job]:
        """Signal cancellation and mark the job cancelled.

        The tool stops at its next :attr:`JobContext.cancelled` check, so a
        tool that never checks keeps running until ``max_duration_seconds``.
        Reporting the job as cancelled immediately is still the honest
        answer to the *caller*: nothing will consume its result.
        """

        job = self.store.get(job_id)
        if job is None or job.terminal:
            return job
        with self._condition:
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
        job.status = JobStatus.CANCELLED
        job.finished_at = time.time()
        job.error = "cancelled"
        if not self.store.put_if_not_terminal(job):
            # Something else reached a terminal state first (the tool
            # returned, or the duration watchdog fired). That verdict wins;
            # report what is actually stored rather than the one we built.
            return self.store.get(job_id)
        self._notify()
        log_event(logger, logging.INFO, "job.cancelled", job_id=job_id)
        return job

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for event in self._cancel_events.values():
                event.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def __enter__(self) -> "JobRunner":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- execution ----------------------------------------------------------
    def _execute(self, job_id: str, tool: BaseTool, arguments: Dict[str, Any]) -> None:
        job = self.store.get(job_id)
        if job is None or job.terminal:
            return
        with self._condition:
            cancel_event = self._cancel_events.get(job_id) or threading.Event()

        now = time.time()
        job.status = JobStatus.RUNNING
        job.started_at = now
        job.heartbeat_at = now
        # A job cancelled between submit() and this worker thread picking it
        # up must stay cancelled -- the plain put() this replaces would have
        # marked it RUNNING again. See BUGS.md #12.
        if not self.store.put_if_not_terminal(job):
            self._notify()
            return
        self._notify()

        watchdog = threading.Timer(
            self.budget.max_duration_seconds, self._time_out, args=(job_id, "max_duration")
        )
        watchdog.daemon = True
        watchdog.start()
        # Stall detection only applies to tools that actually heartbeat; a
        # tool that never reports progress would otherwise be killed for
        # doing exactly what it was written to do.
        heartbeats = _accepts_job_context(tool)
        stall_watch = _StallWatchdog(self, job_id) if heartbeats else None
        if stall_watch is not None:
            stall_watch.start()

        log_event(
            logger,
            logging.INFO,
            "job.started",
            job_id=job_id,
            tool_name=tool.name,
            stall_watched=heartbeats,
        )
        try:
            result = self._invoke(tool, arguments, job_id, cancel_event)
            self._finish(job_id, JobStatus.SUCCEEDED, result=str(result))
        except JobCancelled:
            self._finish(job_id, JobStatus.CANCELLED, error="cancelled")
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "job.failed",
                job_id=job_id,
                tool_name=tool.name,
                error_type=type(exc).__name__,
                exc_info=True,
            )
            self._finish(job_id, JobStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
        finally:
            watchdog.cancel()
            if stall_watch is not None:
                stall_watch.stop()

    def _invoke(
        self,
        tool: BaseTool,
        arguments: Dict[str, Any],
        job_id: str,
        cancel_event: threading.Event,
    ) -> str:
        """Run the tool, handing it a :class:`JobContext` only if it wants one."""

        if _accepts_job_context(tool):
            context = JobContext(
                job_id, cancel_event, lambda progress: self._heartbeat(job_id, progress)
            )
            return tool.run(**arguments, job_context=context)
        return tool.run(**arguments)

    def _heartbeat(self, job_id: str, progress: str) -> None:
        # A partial, store-side update rather than get/mutate/put: the old
        # read-modify-write could carry a pre-cancellation snapshot back
        # into the store and resurrect a job the caller had already been
        # told was cancelled. See BUGS.md #12.
        self.store.heartbeat(job_id, at=time.time(), progress=progress)

    def _time_out(self, job_id: str, reason: str) -> None:
        job = self.store.get(job_id)
        if job is None or job.terminal:
            return
        with self._condition:
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
        job.status = JobStatus.TIMED_OUT
        job.finished_at = time.time()
        job.error = f"timed out ({reason})"
        if not self.store.put_if_not_terminal(job):
            return
        self._notify()
        log_event(
            logger, logging.WARNING, "job.timed_out", job_id=job_id, reason=reason
        )

    def _finish(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        job = self.store.get(job_id)
        if job is None:
            return
        job.status = status
        job.finished_at = time.time()
        job.result = result
        job.error = error
        # A watchdog or a cancel may already have decided this job's fate.
        # Keep that verdict: a tool that returns right after being timed out
        # should not overwrite the timeout the caller was already told
        # about. The check lives in the store so it cannot race the write.
        if not self.store.put_if_not_terminal(job):
            self._notify()
            return
        self._notify()
        log_event(
            logger,
            logging.INFO if status is JobStatus.SUCCEEDED else logging.WARNING,
            "job.completed",
            job_id=job_id,
            status=status.value,
            elapsed_seconds=round(job.elapsed_seconds(), 2),
        )

    def _notify(self) -> None:
        with self._condition:
            self._condition.notify_all()


def _accepts_job_context(tool: BaseTool) -> bool:
    """Whether ``tool.run`` takes a ``job_context`` keyword argument."""

    target = getattr(tool, "_func", None) or tool.run
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    return "job_context" in parameters


class _StallWatchdog(threading.Thread):
    """Gives up on a job that has stopped reporting progress.

    Only meaningful for tools that heartbeat: one that never calls
    :meth:`JobContext.heartbeat` would trip this immediately, so a tool
    that does not accept a ``JobContext`` is not watched at all and is
    bounded by ``max_duration_seconds`` instead.
    """

    def __init__(self, runner: "JobRunner", job_id: str) -> None:
        super().__init__(name=f"job-stall-{job_id}", daemon=True)
        self._runner = runner
        self._job_id = job_id
        self._stop = threading.Event()

    def run(self) -> None:
        interval = min(5.0, self._runner.budget.stall_timeout_seconds / 2)
        while not self._stop.wait(interval):
            job = self._runner.store.get(self._job_id)
            if job is None or job.terminal:
                return
            if job.heartbeat_at is None:
                continue
            if time.time() - job.heartbeat_at > self._runner.budget.stall_timeout_seconds:
                self._runner._time_out(self._job_id, "no_heartbeat")
                return

    def stop(self) -> None:
        self._stop.set()
