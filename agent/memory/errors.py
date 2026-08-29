"""Memory-specific errors."""


class MemoryError(RuntimeError):
    """Base class for durable-memory failures."""


class MemoryConfigurationError(MemoryError):
    """Memory dependencies or index versions are incompatible."""


class MemoryNotFoundError(MemoryError):
    """A requested memory record does not exist."""


class MemoryProtectedError(MemoryError):
    """An automatic lifecycle operation targeted a protected record."""
