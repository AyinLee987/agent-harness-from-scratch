"""Public API — thin facade that wires trigger + state + shared infrastructure.

The :class:`ReActAgent` class is now a lightweight wrapper around
:class:`~agent.trigger.ReActLoop`.  It exists for backward compatibility
and as a convenient single-import entry point.

Example::

    from agent import MockLLM, ReActAgent, ToolRegistry, tool

    @tool
    def calculator(expression: str) -> str:
        '''Evaluate an arithmetic expression.'''
        return str(eval(expression))

    agent = ReActAgent(llm=MockLLM(), tools=ToolRegistry([calculator]))
    print(agent.run("What is 23 times 17?").answer)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .compression import ContextCompressor
from .llm import BaseLLM
from .safety import ToolOutputGuard
from .state.memory import LongTermMemory, ShortTermMemory
from .tools import ToolRegistry
from .trigger.react_loop import DEFAULT_SYSTEM_PROMPT, AgentResult, ReActLoop


class ReActAgent:
    """Thin facade around :class:`~agent.trigger.ReActLoop`.

    All parameters are forwarded directly to :class:`ReActLoop`.
    The ``run(task)`` method returns an :class:`AgentResult`.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 10,
        max_tokens: int = 100_000,
        short_term: Optional[ShortTermMemory] = None,
        long_term: Optional[LongTermMemory] = None,
        compressor: Optional[ContextCompressor] = None,
        output_guard: Optional[ToolOutputGuard] = None,
        compress_at_fraction: float = 0.6,
        max_tool_retries: int = 1,
    ) -> None:
        self._loop = ReActLoop(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            max_steps=max_steps,
            max_tokens=max_tokens,
            short_term=short_term,
            long_term=long_term,
            compressor=compressor,
            output_guard=output_guard,
            compress_at_fraction=compress_at_fraction,
            max_tool_retries=max_tool_retries,
        )
        # Expose attributes for backward compatibility.
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.short_term = self._loop.short_term
        self.long_term = long_term
        self.compressor = compressor
        self.output_guard = output_guard

    def run(self, task: str) -> AgentResult:
        """Run the agent to completion on ``task``."""
        return self._loop.run(task)
