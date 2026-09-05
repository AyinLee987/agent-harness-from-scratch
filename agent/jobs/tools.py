"""The tool-facing side of long-running execution.

Three pieces, deliberately shaped to mirror the delegation tools in
``agent/multi_agent/tools.py`` -- spawn, poll, wait -- because "start
something slow, then find out how it went" is the same problem whether the
slow thing is a Worker or a tool, and a model that has learned one shape
should not have to learn a second.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from ..errors import RecoverableToolError
from ..tools import BaseTool, FunctionTool
from .models import TERMINAL_JOB_STATUSES, Job, JobStatus, SuspendRun
from .runner import JobRunner


class LongRunningTool(BaseTool):
    """Wraps a slow tool so calling it returns a handle instead of blocking.

    The wrapped tool's schema is preserved exactly, so the model calls it
    with the same arguments it always would; only the *return* changes,
    from the result to a handle plus instructions for collecting it. That
    keeps the change invisible to prompt engineering: a tool becomes
    long-running by how it is registered, not by how it is described.
    """

    def __init__(
        self,
        inner: BaseTool,
        runner: JobRunner,
        *,
        name: Optional[str] = None,
    ) -> None:
        self._inner = inner
        self._runner = runner
        self.name = name or inner.name
        self.description = (
            f"{inner.description} "
            f"(Long-running: returns a job_id immediately. Collect the result "
            f"with await_jobs, or poll it with job_status.)"
        )

    def parameters_schema(self) -> Dict[str, Any]:
        return self._inner.parameters_schema()

    def run(self, **kwargs: Any) -> str:
        job = self._runner.submit(self._inner, kwargs)
        return json.dumps(
            {
                "job_id": job.job_id,
                "tool_name": job.tool_name,
                "status": job.status.value,
                "next": (
                    "Call await_jobs with this job_id to collect the result. "
                    "Do not call this tool again with the same arguments -- "
                    "it would return this same job."
                ),
            },
            ensure_ascii=False,
        )


def create_job_tools(
    runner: JobRunner, *, default_wait_seconds: float = 30.0
) -> List[BaseTool]:
    """Build the ``job_status`` / ``await_jobs`` / ``cancel_job`` tool set."""

    def job_status(job_id: str) -> str:
        """Check whether a long-running job has finished, and get its result.

        Args:
            job_id: The id returned when the long-running tool was called.
        """
        job = runner.get(job_id)
        if job is None:
            raise RecoverableToolError(
                f"Unknown job_id {job_id!r}. Check the id from the tool's own reply."
            )
        return json.dumps(job.to_dict(), ensure_ascii=False)

    def await_jobs(job_ids: str, timeout_seconds: float = default_wait_seconds) -> str:
        """Wait for one or more long-running jobs and return their results.

        Args:
            job_ids: One job id, or several separated by commas.
            timeout_seconds: How long to wait before suspending the run.
        """
        ids = [item.strip() for item in str(job_ids).split(",") if item.strip()]
        if not ids:
            raise RecoverableToolError("job_ids must name at least one job.")
        missing = [item for item in ids if runner.get(item) is None]
        if missing:
            raise RecoverableToolError(
                f"Unknown job_id(s): {', '.join(missing)}."
            )

        jobs = runner.await_jobs(ids, timeout_seconds=max(1.0, float(timeout_seconds)))
        pending = [job.job_id for job in jobs if job.status not in TERMINAL_JOB_STATUSES]
        if pending:
            # Not an error: on this path "not finished yet" is the expected
            # case. Suspending hands the run back to the caller with its
            # state intact, to be resumed when the jobs land -- which is the
            # whole reason the work was taken off the call stack.
            raise SuspendRun(pending)
        return json.dumps(
            [job.to_dict() for job in jobs], ensure_ascii=False
        )

    def cancel_job(job_id: str) -> str:
        """Stop a long-running job whose result is no longer needed.

        Args:
            job_id: The id returned when the long-running tool was called.
        """
        job = runner.cancel(job_id)
        if job is None:
            raise RecoverableToolError(f"Unknown job_id {job_id!r}.")
        return json.dumps(job.to_dict(include_result=False), ensure_ascii=False)

    return [
        FunctionTool(job_status, error_policy="recoverable"),
        FunctionTool(await_jobs, error_policy="recoverable"),
        FunctionTool(cancel_job, error_policy="recoverable"),
    ]


def describe_jobs(jobs: Sequence[Job]) -> str:
    """Render a resumed run's job results as one tool observation."""

    return json.dumps([job.to_dict() for job in jobs], ensure_ascii=False)
