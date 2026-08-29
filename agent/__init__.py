"""agent-harness-from-scratch: a production-shaped ReAct agent framework.

Architecture: Trigger Layer (when/how the agent acts) + State Layer
(what the agent knows) + shared infrastructure.

Public surface:

* :class:`~agent.agent.ReActAgent` / :class:`~agent.agent.AgentResult`
* :class:`~agent.state.ExecutionContext`
* :class:`~agent.tools.BaseTool`, :func:`~agent.tools.tool`,
  :class:`~agent.tools.ToolRegistry`
* :class:`~agent.state.ShortTermMemory`, :class:`~agent.state.LongTermMemory`
* :class:`~agent.trigger.StateGraph`
* :class:`~agent.llm.BaseLLM`, :class:`~agent.llm.MockLLM`,
  :class:`~agent.llm.OpenAILLM`, :class:`~agent.llm.BailianLLM`
* :class:`~agent.trigger.AgentGateway`
* :class:`~agent.state.BaseVectorStore`,
  :class:`~agent.state.NumPyVectorStore`,
  :class:`~agent.state.SQLiteVectorStore`
"""

from .agent import AgentResult, ReActAgent
from .compression import CompressionResult, ContextCompressor
from .errors import (
    FatalToolError,
    RecoverableToolError,
    RespondToModel,
    ToolCallError,
)
from .llm import (
    BailianLLM,
    BaseLLM,
    DeepSeekLLM,
    LLMResponse,
    MockLLM,
    OpenAILLM,
    ToolCall,
    Usage,
)
from .multi_agent import (
    AgentRegistry,
    AgentSpec,
    MultiAgentOrchestrator,
    MultiAgentRunResult,
    RunBudget,
    SubagentResult,
    SubagentTask,
    TaskStatus,
    create_leader_tools,
)
from .safety import ScanResult, ToolOutputGuard
from .state import (
    BaseVectorStore,
    ExecutionContext,
    LongTermMemory,
    MemoryRecord,
    NumPyVectorStore,
    ShortTermMemory,
    SQLiteVectorStore,
    Step,
)
from .tools import BaseTool, FunctionTool, ToolRegistry, tool
from .trigger import (
    AgentGateway,
    ConcurrencyGuard,
    RateLimiter,
    ReActLoop,
    RequestQueue,
    StateGraph,
    ToolDispatcher,
)
from .trigger.gateway import (
    ConcurrencyLimitExceeded,
    GatewayError,
    GatewayResult,
    QueueTimeout,
    RateLimitExceeded,
)

__version__ = "0.2.0"

__all__ = [
    # Agent
    "ReActAgent",
    "AgentResult",
    # Trigger layer
    "StateGraph",
    "ReActLoop",
    "ToolDispatcher",
    "AgentGateway",
    "ConcurrencyGuard",
    "RateLimiter",
    "RequestQueue",
    "GatewayError",
    "GatewayResult",
    "RateLimitExceeded",
    "ConcurrencyLimitExceeded",
    "QueueTimeout",
    # Multi-agent
    "AgentRegistry",
    "AgentSpec",
    "MultiAgentOrchestrator",
    "MultiAgentRunResult",
    "RunBudget",
    "SubagentResult",
    "SubagentTask",
    "TaskStatus",
    "create_leader_tools",
    # State layer
    "ExecutionContext",
    "Step",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryRecord",
    "BaseVectorStore",
    "NumPyVectorStore",
    "SQLiteVectorStore",
    # LLM
    "BaseLLM",
    "LLMResponse",
    "MockLLM",
    "OpenAILLM",
    "DeepSeekLLM",
    "BailianLLM",
    "ToolCall",
    "Usage",
    # Compression & safety
    "ContextCompressor",
    "CompressionResult",
    "ToolOutputGuard",
    "ScanResult",
    # Tools
    "ToolCallError",
    "RecoverableToolError",
    "RespondToModel",
    "FatalToolError",
    "BaseTool",
    "FunctionTool",
    "ToolRegistry",
    "tool",
]
