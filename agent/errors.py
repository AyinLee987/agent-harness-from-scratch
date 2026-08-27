"""Error contract shared by tool implementations and the agent loop.

Tool authors classify failures at the operation that can fail.  Recoverable
errors are returned to the model as tool observations; fatal errors stop the
current run immediately.  Unexpected, unclassified exceptions are treated as
fatal by the dispatcher.
"""


class ToolCallError(RuntimeError):
    """Base class for deliberately classified tool-call failures."""


class RecoverableToolError(ToolCallError):
    """A failure the model can address by changing its next action."""


class FatalToolError(ToolCallError):
    """A failure the model cannot repair; the current run must stop."""


# Codex terminology, provided as an expressive alias for tool authors.
RespondToModel = RecoverableToolError
