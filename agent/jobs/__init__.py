"""Long-running tool execution: handles, durable jobs, suspend and resume.

A tool that takes half an hour cannot block the agent's call stack. This
package is the answer to that: a slow tool returns a ``job_id``
immediately, the work continues on the runner's own threads, and the run
suspends -- with a resumable checkpoint -- rather than sitting on a thread
waiting.

See :mod:`agent.jobs.models` for the one-sentence rule that shapes all of it.
"""

from .models import (
    TERMINAL_JOB_STATUSES,
    Job,
    JobStatus,
    SuspendRun,
)
from .runner import JobBudget, JobCancelled, JobContext, JobRunner, fingerprint
from .store import InMemoryJobStore, JobStore, SQLiteJobStore
from .tools import LongRunningTool, create_job_tools, describe_jobs

__all__ = [
    "TERMINAL_JOB_STATUSES",
    "InMemoryJobStore",
    "Job",
    "JobBudget",
    "JobCancelled",
    "JobContext",
    "JobRunner",
    "JobStatus",
    "JobStore",
    "LongRunningTool",
    "SQLiteJobStore",
    "SuspendRun",
    "create_job_tools",
    "describe_jobs",
    "fingerprint",
]
