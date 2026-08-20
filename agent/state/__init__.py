"""State Layer — manages WHAT the agent knows.

Components:
    - ExecutionContext: messages, steps, budget, run_id — all mutable run state
    - ShortTermMemory: sliding window + summarization fallback
    - LongTermMemory: persistent-capable vector recall
    - VectorStore: pluggable backends (NumPy / SQLite / future FAISS, Qdrant)
"""

from .context import ExecutionContext, Step
from .memory import LongTermMemory, MemoryRecord, ShortTermMemory
from .store import BaseVectorStore, NumPyVectorStore, SQLiteVectorStore

__all__ = [
    "BaseVectorStore",
    "ExecutionContext",
    "LongTermMemory",
    "MemoryRecord",
    "NumPyVectorStore",
    "ShortTermMemory",
    "SQLiteVectorStore",
    "Step",
]
