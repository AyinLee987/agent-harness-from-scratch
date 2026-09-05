"""Durable record of long-running tool executions.

Persistence is the whole point rather than a nice-to-have: an in-process
dict makes a half-hour job a half-hour window in which a deploy, a crash,
or an OOM silently loses work that has already been paid for. A restarted
process must be able to look up what it had running and report honestly on
it, which is exactly the difference between a request-scoped agent and a
durable one.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .models import TERMINAL_JOB_STATUSES, Job, JobStatus


class JobStore(Protocol):
    def put(self, job: Job) -> None: ...

    def put_if_not_terminal(self, job: Job) -> bool: ...

    def heartbeat(self, job_id: str, *, at: float, progress: str = "") -> bool: ...

    def get(self, job_id: str) -> Optional[Job]: ...

    def find_reusable(self, fingerprint: str, *, ttl_seconds: float) -> Optional[Job]: ...

    def list_unfinished(self) -> List[Job]: ...

    def delete(self, job_id: str) -> bool: ...


class InMemoryJobStore:
    """Zero-dependency reference implementation, and what tests use."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.RLock()

    def put(self, job: Job) -> None:
        with self._lock:
            self._jobs[job.job_id] = copy.deepcopy(job)

    def put_if_not_terminal(self, job: Job) -> bool:
        """Write ``job`` only if the stored row is not already terminal.

        The compare and the write happen under one lock, which ``get()``
        then ``put()`` in the caller cannot do: a cancel landing between
        those two calls was silently overwritten by the caller's stale
        snapshot, turning a job the caller had already been told was
        CANCELLED back into RUNNING. See BUGS.md #12.

        Returns whether the write happened.
        """

        with self._lock:
            current = self._jobs.get(job.job_id)
            if current is not None and current.terminal:
                return False
            self._jobs[job.job_id] = copy.deepcopy(job)
            return True

    def heartbeat(self, job_id: str, *, at: float, progress: str = "") -> bool:
        """Update only progress fields, and only while the job is running.

        A partial update rather than a whole-row write, so a heartbeat can
        never carry a stale copy of any other field back into the store.
        """

        with self._lock:
            current = self._jobs.get(job_id)
            if current is None or current.terminal:
                return False
            current.heartbeat_at = at
            if progress:
                current.progress = progress[:500]
            return True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job is not None else None

    def find_reusable(self, fingerprint: str, *, ttl_seconds: float) -> Optional[Job]:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            candidates = [
                job
                for job in self._jobs.values()
                if job.fingerprint == fingerprint and job.created_at >= cutoff
                and job.status is not JobStatus.FAILED
            ]
        if not candidates:
            return None
        return copy.deepcopy(max(candidates, key=lambda job: job.created_at))

    def list_unfinished(self) -> List[Job]:
        with self._lock:
            return copy.deepcopy(
                [job for job in self._jobs.values() if job.status not in TERMINAL_JOB_STATUSES]
            )

    def delete(self, job_id: str) -> bool:
        with self._lock:
            return self._jobs.pop(job_id, None) is not None


class SQLiteJobStore:
    """SQLite source of truth using a version-tolerant JSON job payload."""

    def __init__(self, db_path: str) -> None:
        if db_path == ":memory:":
            self.db_path = db_path
        else:
            path = Path(db_path).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = str(path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    job_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_fingerprint "
                "ON jobs(fingerprint, created_at)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)"
            )
            self._conn.commit()

    def put(self, job: Job) -> None:
        payload = json.dumps(job.to_record(), ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET "
                "fingerprint=excluded.fingerprint, run_id=excluded.run_id, "
                "tool_name=excluded.tool_name, status=excluded.status, "
                "created_at=excluded.created_at, job_json=excluded.job_json",
                (
                    job.job_id,
                    job.fingerprint,
                    job.run_id,
                    job.tool_name,
                    job.status.value,
                    job.created_at,
                    payload,
                ),
            )
            self._conn.commit()

    def put_if_not_terminal(self, job: Job) -> bool:
        """Write ``job`` only if the stored row is not already terminal.

        One statement, so the check and the write cannot interleave --
        see :meth:`InMemoryJobStore.put_if_not_terminal` and BUGS.md #12.
        """

        payload = json.dumps(job.to_record(), ensure_ascii=False)
        terminal = [status.value for status in TERMINAL_JOB_STATUSES]
        placeholders = ", ".join("?" for _ in terminal)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE jobs SET fingerprint=?, run_id=?, tool_name=?, status=?, "
                f"created_at=?, job_json=? "
                f"WHERE job_id=? AND status NOT IN ({placeholders})",
                (
                    job.fingerprint,
                    job.run_id,
                    job.tool_name,
                    job.status.value,
                    job.created_at,
                    payload,
                    job.job_id,
                    *terminal,
                ),
            )
            if cursor.rowcount == 0:
                # Either the row is terminal, or it does not exist yet.
                exists = self._conn.execute(
                    "SELECT 1 FROM jobs WHERE job_id = ?", (job.job_id,)
                ).fetchone()
                if exists:
                    self._conn.commit()
                    return False
                self._conn.execute(
                    "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        job.job_id,
                        job.fingerprint,
                        job.run_id,
                        job.tool_name,
                        job.status.value,
                        job.created_at,
                        payload,
                    ),
                )
            self._conn.commit()
            return True

    def heartbeat(self, job_id: str, *, at: float, progress: str = "") -> bool:
        """Update only progress fields, and only while the job is running."""

        terminal = [status.value for status in TERMINAL_JOB_STATUSES]
        placeholders = ", ".join("?" for _ in terminal)
        with self._lock:
            row = self._conn.execute(
                f"SELECT job_json FROM jobs WHERE job_id = ? "
                f"AND status NOT IN ({placeholders})",
                (job_id, *terminal),
            ).fetchone()
            if row is None:
                return False
            job = Job.from_record(json.loads(row["job_json"]))
            job.heartbeat_at = at
            if progress:
                job.progress = progress[:500]
            self._conn.execute(
                f"UPDATE jobs SET job_json=? WHERE job_id=? "
                f"AND status NOT IN ({placeholders})",
                (json.dumps(job.to_record(), ensure_ascii=False), job_id, *terminal),
            )
            self._conn.commit()
            return True

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            row = self._conn.execute(
                "SELECT job_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return Job.from_record(json.loads(row["job_json"])) if row else None

    def find_reusable(self, fingerprint: str, *, ttl_seconds: float) -> Optional[Job]:
        cutoff = time.time() - ttl_seconds
        with self._lock:
            row = self._conn.execute(
                "SELECT job_json FROM jobs WHERE fingerprint = ? AND created_at >= ? "
                "AND status != ? ORDER BY created_at DESC LIMIT 1",
                (fingerprint, cutoff, JobStatus.FAILED.value),
            ).fetchone()
        return Job.from_record(json.loads(row["job_json"])) if row else None

    def list_unfinished(self) -> List[Job]:
        terminal = [status.value for status in TERMINAL_JOB_STATUSES]
        placeholders = ", ".join("?" for _ in terminal)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT job_json FROM jobs WHERE status NOT IN ({placeholders}) "
                f"ORDER BY created_at",
                terminal,
            ).fetchall()
        return [Job.from_record(json.loads(row["job_json"])) for row in rows]

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
