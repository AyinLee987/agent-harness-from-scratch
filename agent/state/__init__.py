"""State Layer — manages WHAT the agent knows.

Components:
    - ExecutionContext: messages, steps, budget, run_id — all mutable run state
    - ShortTermMemory: sliding window + summarization fallback
    - LongTermMemory: persistent-capable vector recall
    - VectorStore: pluggable backends (NumPy / SQLite / Chroma / future FAISS, Qdrant)
    - CheckpointStore: durable snapshots of runs suspended on long-running jobs
"""

from .checkpoints import (
    CheckpointStore,
    InMemoryCheckpointStore,
    RunCheckpoint,
    SQLiteCheckpointStore,
)
from .chroma_store import ChromaVectorStore, VectorStoreConfigurationError
from .context import ExecutionContext, Step
from .memory import LongTermMemory, MemoryRecord, ShortTermMemory
from .store import BaseVectorStore, NumPyVectorStore, SQLiteVectorStore

__all__ = [
    "BaseVectorStore",
    "CheckpointStore",
    "InMemoryCheckpointStore",
    "RunCheckpoint",
    "SQLiteCheckpointStore",
    "ChromaVectorStore",
    "ExecutionContext",
    "LongTermMemory",
    "MemoryRecord",
    "NumPyVectorStore",
    "ShortTermMemory",
    "SQLiteVectorStore",
    "Step",
    "VectorStoreConfigurationError",
]
