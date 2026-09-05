"""Error contract shared by tool implementations and the agent loop.

Tool authors classify failures at the operation that can fail.  Recoverable
errors are returned to the model as tool observations; fatal errors stop the
current run immediately.  Unexpected, unclassified exceptions are treated as
fatal by the dispatcher.

Separately from failures, a tool may raise a :class:`ControlSignal` to
change what the *loop* does next.  The two hierarchies are deliberately
disjoint -- see :class:`ControlSignal`.
"""


class ToolCallError(RuntimeError):
    """Base class for deliberately classified tool-call failures."""


class RecoverableToolError(ToolCallError):
    """A failure the model can address by changing its next action."""


class FatalToolError(ToolCallError):
    """A failure the model cannot repair; the current run must stop."""


# Codex terminology, provided as an expressive alias for tool authors.
RespondToModel = RecoverableToolError


class ControlSignal(Exception):
    """A tool telling the loop to do something other than continue.

    Explicitly **not** a :class:`ToolCallError`, and not a third failure
    tier: nothing has gone wrong when one of these is raised.  The tool
    taxonomy above answers "how bad was this failure"; this answers "what
    should the loop do instead of taking another step", and conflating them
    would make the two-tier failure contract stop meaning anything.

    Because it is a control decision rather than a failure, it propagates
    untouched through :class:`~agent.tools.FunctionTool` and
    :class:`~agent.trigger.ToolDispatcher`, which otherwise classify every
    exception they see.  The concrete signal today is
    :class:`~agent.jobs.SuspendRun`.
    """
