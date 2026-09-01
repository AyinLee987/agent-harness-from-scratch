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
from .context import ContextProvider
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
from .local_tools import (
    ListFilesTool, LocalToolConfig, ReadFileTool, RunCommandTool, WriteFileTool,
    create_local_tools,
)
from .memory import (
    DefaultMemoryPolicy,
    EmbeddingProvider,
    ExplicitRequestMemoryExtractor,
    InMemoryMemoryRepository,
    InMemorySessionStore,
    InMemoryVectorIndex,
    LLMEmbeddingProvider,
    MemoryCandidate,
    MemoryConfigurationError,
    MemoryDecision,
    MemoryError,
    MemoryKind,
    MemoryManager,
    MemoryNotFoundError,
    MemoryProtectedError,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    NoopMemoryExtractor,
    OpenAICompatibleEmbeddingProvider,
    RetentionPolicy,
    RunCompletedEvent,
    SQLiteMemoryRepository,
    SQLiteSessionStore,
    Sensitivity,
    SessionContextProvider,
    SessionMemoryStore,
    SummarySnapshot,
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
from .observability import (
    JsonFormatter,
    RequestLoggingMiddleware,
    TextFormatter,
    bind_log_context,
    configure_logging,
    current_log_context,
    get_logger,
    log_event,
    sanitize,
)
from .rag import (
    BM25Retriever, CallableReranker, Chunk, Citation, CitationCounter, DenseRetriever, Document,
    DocumentStatus, Evidence, EvidenceBundle, EvidenceConflict, EvidenceStatus,
    HeuristicReranker, InMemoryRAGRepository, LLMQueryDecomposer, MedicalParentChildChunker,
    MedicalQueryPlanner, QueryDecomposer, QueryDecomposition, RAGConfig, RAGContextProvider,
    RAGIngestionService, RAGPipeline, RetrievalFilters, SQLiteRAGRepository,
    create_rag_search_tool, format_evidence_context,
)
from .safety import ScanResult, ToolOutputGuard
from .state import (
    BaseVectorStore,
    ExecutionContext,
    LongTermMemory,
    MemoryRecord as LegacyMemoryRecord,
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
    "ContextProvider",
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
    # Durable memory
    "MemoryManager",
    "MemoryCandidate",
    "MemoryRecord",
    "LegacyMemoryRecord",
    "MemorySearchResult",
    "MemoryKind",
    "MemoryStatus",
    "MemoryDecision",
    "RetentionPolicy",
    "Sensitivity",
    "RunCompletedEvent",
    "EmbeddingProvider",
    "LLMEmbeddingProvider",
    "InMemoryMemoryRepository",
    "SQLiteMemoryRepository",
    "InMemoryVectorIndex",
    "InMemorySessionStore",
    "SQLiteSessionStore",
    "SessionContextProvider",
    "SessionMemoryStore",
    "DefaultMemoryPolicy",
    "NoopMemoryExtractor",
    "OpenAICompatibleEmbeddingProvider",
    "ExplicitRequestMemoryExtractor",
    "SummarySnapshot",
    "MemoryError",
    "MemoryConfigurationError",
    "MemoryNotFoundError",
    "MemoryProtectedError",
    # RAG
    "Document", "DocumentStatus", "Chunk", "Citation", "Evidence",
    "EvidenceBundle", "EvidenceConflict", "EvidenceStatus", "RetrievalFilters",
    "MedicalParentChildChunker", "MedicalQueryPlanner", "RAGIngestionService",
    "InMemoryRAGRepository", "SQLiteRAGRepository", "BM25Retriever",
    "DenseRetriever", "HeuristicReranker", "CallableReranker", "RAGConfig",
    "RAGPipeline", "RAGContextProvider", "create_rag_search_tool",
    "format_evidence_context", "QueryDecomposer", "QueryDecomposition",
    "LLMQueryDecomposer", "CitationCounter",
    # Observability
    "configure_logging",
    "bind_log_context",
    "current_log_context",
    "get_logger",
    "log_event",
    "sanitize",
    "JsonFormatter",
    "TextFormatter",
    "RequestLoggingMiddleware",
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
    # Local workspace tools
    "LocalToolConfig", "ReadFileTool", "WriteFileTool", "ListFilesTool",
    "RunCommandTool", "create_local_tools",
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
