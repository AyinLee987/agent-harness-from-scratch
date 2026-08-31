"""A verbose-description variant of the 50-tool kit.

Isolates a different axis from ``tool_scaling_kit.py``: not "how many
tools", but "how long is each tool's description" -- the original framing
of the interview question this whole experiment answers ("tool 太多导致
description 太长, 导致模型调用工具准确率降低"). Tool *count* stays fixed
at 50 and every name/parameter/behavior is identical to the concise kit;
only ``description`` is bloated, by wrapping each concise tool in
:class:`VerboseTool`.

The padding is deliberately generic boilerplate (safety notes, usage
guidance, performance claims) repeated near-verbatim across all 50 tools --
that is itself realistic: this is exactly how tool docs bloat in practice
(a compliance/safety paragraph gets pasted into every tool's docstring).
The practical effect on the model is the opposite of helpful information
density: the distinguishing detail for each tool is now a small island in a
sea of identical text, which is the actual mechanism by which verbose
descriptions hurt selection accuracy -- not context size alone.

Run ``python examples/tool_scaling_verbose_kit.py`` to print the size
delta (chars / estimated tokens) between the concise and verbose schemas.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`
sys.path.insert(0, _HERE)  # examples dir -> `import tool_scaling_kit`

from agent import BaseTool  # noqa: E402

from tool_scaling_kit import ALL_TOOLS  # noqa: E402

_CATEGORY_BY_PREFIX = {
    "math_": "arithmetic and statistics",
    "text_": "string manipulation",
    "date_": "date and time",
    "convert_": "unit conversion",
    "data_": "encoding and data-utility",
}


def _category_for(name: str) -> str:
    for prefix, label in _CATEGORY_BY_PREFIX.items():
        if name.startswith(prefix):
            return label
    return "general-purpose"


# Generic boilerplate, deliberately near-identical across all 50 tools --
# this is what actually erodes the signal that distinguishes one tool's
# description from another's, not sheer character count by itself.
_BOILERPLATE = (
    "This tool is part of the {category} utility group within the agent's "
    "tool catalog and is safe to invoke in any context: it has no side "
    "effects, performs no network I/O, writes nothing to disk, and does not "
    "mutate any external or shared state. It accepts the parameters "
    "documented below, all of which must be supplied with the exact types "
    "shown in the schema, and it returns a single plain-text string "
    "observation that the agent should treat as the authoritative result of "
    "the operation. Typical use cases include automated data-processing "
    "pipelines, interactive assistant queries where a user asks a natural- "
    "language question that maps onto this operation, and batch or "
    "scripted workflows that chain multiple tool calls together. If the "
    "supplied input is malformed, out of range, or otherwise cannot be "
    "processed, the tool raises a descriptive error message rather than "
    "failing silently or returning a misleading default value, so callers "
    "should inspect the returned text for an error indication before "
    "treating the result as valid and using it in a subsequent step. This "
    "tool completes in effectively constant time relative to the size of "
    "its input, aside from parameters that accept lists or delimited "
    "strings of values, which scale linearly with the number of elements "
    "supplied. Prefer this tool over performing the equivalent computation "
    "manually whenever the available input already matches its expected "
    "format, since it guarantees deterministic, reproducible output across "
    "repeated invocations with the same arguments, and its behavior is "
    "covered by the project's automated test suite."
)


class VerboseTool(BaseTool):
    """Wraps a concise tool, replacing only its description with a bloated
    version. Name, parameter schema, and ``run()`` behavior are untouched --
    this isolates description length as the sole variable under test."""

    def __init__(self, inner: BaseTool) -> None:
        self._inner = inner
        self.name = inner.name
        summary = inner.description or inner.name
        boilerplate = _BOILERPLATE.format(category=_category_for(inner.name))
        self.description = f"{summary}\n\n{boilerplate}"

    def parameters_schema(self) -> Dict[str, Any]:
        return self._inner.parameters_schema()

    def run(self, **kwargs: Any) -> str:
        return self._inner.run(**kwargs)


ALL_TOOLS_VERBOSE = [VerboseTool(t) for t in ALL_TOOLS]

assert len(ALL_TOOLS_VERBOSE) == 50
assert {t.name for t in ALL_TOOLS_VERBOSE} == {t.name for t in ALL_TOOLS}


def _schema_chars(tools) -> int:
    import json
    return len(json.dumps([t.to_schema() for t in tools]))


def main() -> None:
    concise_chars = _schema_chars(ALL_TOOLS)
    verbose_chars = _schema_chars(ALL_TOOLS_VERBOSE)
    print(f"Concise 50-tool schema:  {concise_chars:>7} chars  (~{concise_chars // 4} tokens)")
    print(f"Verbose 50-tool schema:  {verbose_chars:>7} chars  (~{verbose_chars // 4} tokens)")
    print(f"Growth: {verbose_chars / concise_chars:.1f}x")


if __name__ == "__main__":
    main()
