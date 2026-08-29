"""Registry of named Worker roles and factories."""

from __future__ import annotations

import threading
from typing import Callable, Dict, List

from ..agent import ReActAgent
from .models import AgentSpec

AgentFactory = Callable[[], ReActAgent]


class AgentRegistry:
    """Maps a stable role name to a factory that creates a fresh Agent."""

    def __init__(self) -> None:
        self._specs: Dict[str, AgentSpec] = {}
        self._factories: Dict[str, AgentFactory] = {}
        self._lock = threading.RLock()

    def register(self, spec: AgentSpec, factory: AgentFactory) -> AgentSpec:
        if not callable(factory):
            raise TypeError("Agent factory must be callable.")
        with self._lock:
            if spec.name in self._specs:
                raise ValueError(f"Agent role {spec.name!r} is already registered.")
            self._specs[spec.name] = spec
            self._factories[spec.name] = factory
        return spec

    def __contains__(self, name: str) -> bool:
        with self._lock:
            return name in self._specs

    def names(self) -> List[str]:
        with self._lock:
            return list(self._specs)

    def specs(self) -> List[AgentSpec]:
        with self._lock:
            return list(self._specs.values())

    def get(self, name: str) -> AgentSpec | None:
        with self._lock:
            return self._specs.get(name)

    def create(self, name: str) -> ReActAgent:
        with self._lock:
            factory = self._factories.get(name)
        if factory is None:
            raise KeyError(name)
        agent = factory()
        if not isinstance(agent, ReActAgent):
            raise TypeError(
                f"Factory for {name!r} must return ReActAgent, "
                f"got {type(agent).__name__}."
            )
        return agent

    def role_summary(self) -> str:
        with self._lock:
            return "; ".join(
                f"{spec.name}: {spec.description}" for spec in self._specs.values()
            )

