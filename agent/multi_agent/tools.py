"""Leader-facing tools backed by :class:`MultiAgentOrchestrator`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

from ..tools import BaseTool, tool

if TYPE_CHECKING:
    from .orchestrator import MultiAgentOrchestrator


def create_leader_tools(orchestrator: "MultiAgentOrchestrator") -> List[BaseTool]:
    """Create tools bound to one orchestrator.

    The tools resolve the active root run from thread-local state, so the same
    orchestrator may serve independent leader runs on different threads.
    """

    roles = orchestrator.registry.role_summary() or "(no worker roles registered)"

    @tool
    def spawn_subagent(role: str, task: str) -> str:
        """Start a Worker task and immediately return its task id.

        Available roles: ROLE_SUMMARY

        Args:
            role: Registered Worker role to use.
            task: Self-contained instruction for the Worker.
        """

        return json.dumps(
            orchestrator.spawn_subagent(role, task), ensure_ascii=False
        )

    spawn_subagent.description = spawn_subagent.description.replace("ROLE_SUMMARY", roles)

    @tool
    def get_subagent_status(task_id: str) -> str:
        """Inspect one previously spawned Worker task.

        Args:
            task_id: Task id returned by spawn_subagent.
        """

        return json.dumps(
            orchestrator.get_subagent_status(task_id), ensure_ascii=False
        )

    @tool
    def wait_subagents(task_ids: list, timeout_seconds: float = 60.0) -> str:
        """Wait until all selected Worker tasks finish and return their results.

        Args:
            task_ids: Task ids returned by spawn_subagent.
            timeout_seconds: Maximum time to wait before returning control.
        """

        return json.dumps(
            orchestrator.wait_subagents(task_ids, timeout_seconds),
            ensure_ascii=False,
        )

    @tool
    def cancel_subagent(task_id: str) -> str:
        """Request cancellation of a pending or running Worker task.

        Args:
            task_id: Task id returned by spawn_subagent.
        """

        return json.dumps(
            orchestrator.cancel_subagent(task_id), ensure_ascii=False
        )

    return [
        spawn_subagent,
        get_subagent_status,
        wait_subagents,
        cancel_subagent,
    ]

