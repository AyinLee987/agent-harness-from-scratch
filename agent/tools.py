"""Typed tool abstraction.

Two ways to define a tool:

1. Subclass :class:`BaseTool` for full control.
2. Decorate a plain function with :func:`tool` -- its signature and docstring are
   introspected to auto-generate the JSON schema used for tool-calling.

A :class:`ToolRegistry` collects tools and emits the schema list that the LLM
client expects, and dispatches calls by name.
"""

from __future__ import annotations

import collections.abc
import inspect
import types
import typing
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, get_args, get_origin, get_type_hints

from .errors import ControlSignal, FatalToolError, RecoverableToolError, ToolCallError


# Map Python types to JSON-schema primitive types.
_PY_TO_JSON: Dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


_SEQUENCE_ORIGINS = (
    list,
    set,
    frozenset,
    tuple,
    collections.abc.Sequence,
    collections.abc.Set,
    collections.abc.MutableSequence,
)
_MAPPING_ORIGINS = (dict, collections.abc.Mapping, collections.abc.MutableMapping)


def _json_schema(annotation: Any) -> Dict[str, Any]:
    """Best-effort JSON-schema fragment for one parameter annotation.

    A container keeps its *container* type and describes its members
    separately. The earlier version treated everything with a
    ``get_origin()`` as a Union and returned the first type argument, so
    ``list[int]`` was advertised to the model as ``integer`` and
    ``dict[str, int]`` as ``string`` -- a model following that schema sends
    a bare scalar where the tool binds a container, which surfaces as a
    confusing ``TypeError`` from inside the tool rather than as a schema
    problem. See BUGS.md #9.
    """

    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    if origin is None:
        return {"type": _PY_TO_JSON.get(annotation, "string")}

    args = get_args(annotation)

    # Literal["a", "b"] -> an enum, which is the single most useful thing a
    # tool schema can tell a model: it removes the guess entirely.
    if origin is Literal:
        values = [value for value in args]
        member = _PY_TO_JSON.get(type(values[0]), "string") if values else "string"
        return {"type": member, "enum": values}

    # Optional[X] / Union[...] and the 3.10+ ``X | None`` spelling, which has
    # a different origin (types.UnionType) than typing.Union.
    if origin is typing.Union or origin is getattr(types, "UnionType", None):
        members = [arg for arg in args if arg is not type(None)]
        if not members:
            return {"type": "string"}
        schema = _json_schema(members[0])
        if len(members) > 1:
            # Not expressible as one type; the first member is the honest
            # best guess and stays consistent with the pre-fix behaviour.
            return schema
        return schema

    if origin in _SEQUENCE_ORIGINS:
        # tuple[int, str] is heterogeneous -- describing it as an array of
        # ints would be a lie, so it gets an untyped array instead.
        element_args = [arg for arg in args if arg is not Ellipsis]
        homogeneous = len(set(element_args)) == 1 if element_args else False
        schema: Dict[str, Any] = {"type": "array"}
        if homogeneous:
            schema["items"] = _json_schema(element_args[0])
        return schema

    if origin in _MAPPING_ORIGINS:
        schema = {"type": "object"}
        if len(args) == 2:
            schema["additionalProperties"] = _json_schema(args[1])
        return schema

    # An unrecognized generic (a custom Generic, say). Fall back to the
    # origin's own primitive rather than to its type parameters.
    return {"type": _PY_TO_JSON.get(origin, "string")}


def _parse_docstring(doc: Optional[str]) -> tuple[str, Dict[str, str]]:
    """Split a docstring into a summary and per-parameter descriptions.

    Recognizes a simple ``Args:`` block with ``name: description`` lines.
    """

    if not doc:
        return "", {}
    lines = [ln.rstrip() for ln in doc.strip().splitlines()]
    summary_parts: List[str] = []
    params: Dict[str, str] = {}
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "params:", "parameters:"):
            in_args = True
            continue
        if in_args:
            if ":" in stripped and stripped:
                name, _, desc = stripped.partition(":")
                params[name.strip()] = desc.strip()
            elif not stripped:
                in_args = False
        else:
            summary_parts.append(stripped)
    return " ".join(p for p in summary_parts if p).strip(), params


class BaseTool(ABC):
    """Abstract base class for tools.

    Concrete tools must set :attr:`name`/:attr:`description`, implement
    :meth:`run`, and provide a JSON-schema ``parameters`` block via
    :meth:`parameters_schema`.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool and return an observation string."""

    def parameters_schema(self) -> Dict[str, Any]:
        """Return the JSON-schema ``parameters`` object for this tool."""

        return {"type": "object", "properties": {}, "required": []}

    def to_schema(self) -> Dict[str, Any]:
        """Return the OpenAI-style ``function`` tool schema."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }


class FunctionTool(BaseTool):
    """A :class:`BaseTool` backed by a plain Python function.

    Produced by the :func:`tool` decorator; the function signature and docstring
    drive the auto-generated JSON schema.
    """

    def __init__(
        self,
        func: Callable[..., Any],
        name: Optional[str] = None,
        error_policy: Literal["fatal", "recoverable"] = "fatal",
    ) -> None:
        if error_policy not in ("fatal", "recoverable"):
            raise ValueError("error_policy must be 'fatal' or 'recoverable'.")
        self._func = func
        self.name = name or func.__name__
        self.error_policy = error_policy
        summary, param_docs = _parse_docstring(func.__doc__)
        self.description = summary or self.name
        self._param_docs = param_docs
        self._signature = inspect.signature(func)
        try:
            self._hints = get_type_hints(func)
        except Exception:  # pragma: no cover - exotic annotations
            self._hints = {}

    def parameters_schema(self) -> Dict[str, Any]:
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for pname, param in self._signature.parameters.items():
            if pname == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = self._hints.get(pname, param.annotation)
            prop: Dict[str, Any] = _json_schema(annotation)
            if pname in self._param_docs:
                prop["description"] = self._param_docs[pname]
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {"type": "object", "properties": properties, "required": required}

    def run(self, **kwargs: Any) -> str:
        # Bind separately so malformed model arguments remain recoverable even
        # when the tool's own unexpected-error policy is fatal.
        self._signature.bind(**kwargs)
        try:
            return str(self._func(**kwargs))
        except ControlSignal:
            # A control decision, not a failure -- classifying it would turn
            # "suspend this run" into "this tool errored". See errors.py.
            raise
        except ToolCallError:
            raise
        except Exception as exc:
            error_type = (
                RecoverableToolError
                if self.error_policy == "recoverable"
                else FatalToolError
            )
            raise error_type(str(exc)) from exc


def tool(
    name_or_func: Any = None,
    *,
    error_policy: Literal["fatal", "recoverable"] = "fatal",
) -> Any:
    """Decorator that turns a function into a :class:`FunctionTool`.

    Usage::

        @tool
        def calculator(expression: str) -> str:
            '''Evaluate an arithmetic expression.'''
            ...

        @tool("web_search")
        def search(query: str) -> str:
            ...
    """

    if callable(name_or_func):
        return FunctionTool(name_or_func, error_policy=error_policy)

    def decorator(func: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(func, name=name_or_func, error_policy=error_policy)

    return decorator


class ToolRegistry:
    """Holds tools and provides schema export + dispatch."""

    def __init__(self, tools: Optional[List[BaseTool]] = None) -> None:
        self._tools: Dict[str, BaseTool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, t: BaseTool) -> BaseTool:
        if not t.name:
            raise ValueError("Tool must have a non-empty name.")
        if t.name in self._tools:
            raise ValueError(f"Tool {t.name!r} is already registered.")
        self._tools[t.name] = t
        return t

    def register_many(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Register multiple tools, rejecting any name collision."""

        registered: List[BaseTool] = []
        for item in tools:
            registered.append(self.register(item))
        return registered

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def schemas(self) -> List[Dict[str, Any]]:
        """Return the list of tool schemas for the LLM client."""

        return [t.to_schema() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute the named tool. Raises :class:`KeyError` if unknown."""

        if name not in self._tools:
            raise KeyError(name)
        return self._tools[name].run(**arguments)
