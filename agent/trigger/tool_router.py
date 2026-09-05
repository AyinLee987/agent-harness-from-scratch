"""Narrow the tool schemas offered to the model, one think step at a time.

Every tool a registry holds is serialized into the prompt on *every* LLM
call. `examples/tool_scaling_kit.py` measured what that costs: a flat
100-tool registry serializes to 30,630 characters of schema, and
exact-sequence accuracy on 5-step chains fell to 57%. The standard fix --
and the one `examples/hierarchical_agent_kit.py` measured at 67% -- is to
stop showing the model tools it has no use for.

This module does that by *retrieval* rather than by hand-built specialist
groups: the tool descriptions are a small corpus, the task is a query, and
the same BM25 scorer and tokenizer the document retriever uses (see
``agent/rag/retrieval.py``) pick the top-k. That reuse is the point --
a tool description and a corpus chunk get segmented identically, including
the Chinese bigram handling, instead of this module growing its own
half-right text pipeline.

Three rules keep a narrowed tool set from becoming a *broken* one, since
the failure mode of over-filtering is an agent that cannot finish rather
than one that is merely slow:

* **Below ``min_tools``, selection is a no-op.** Filtering six tools saves
  nothing worth the risk of hiding the right one.
* **A tool already used in this run is never hidden.** Dropping a tool
  mid-chain -- after the model has seen its output and is reasoning about
  calling it again -- produces a confusing "unknown tool" retry loop.
* **Pinned tools are always offered.** Delegation
  (``spawn_subagent``/``wait_subagents``) and any other control-plane tool
  is not something the task text will ever lexically match, but removing
  it changes what the agent is structurally able to do.

Two known limitations, stated plainly rather than left to be discovered:

* **Selection sees a window, not the whole run.** It matches on the task
  plus the last few steps, so a tool that only becomes relevant from an
  observation several steps back can fall out of scope. That is why
  :meth:`LexicalToolSelector.select` takes the recent trajectory at all,
  and why ``min_tools`` defaults high enough that small registries never
  reach this path.
* **Lexical matching is same-language matching.** A Chinese question
  against English tool descriptions shares no terms and scores zero across
  the board -- measured, not hypothetical: on this repo's own 56-tool live
  registry, ``"23 乘以 17 等于多少"`` scores 0.0 on every tool while
  ``"multiply 23 by 17"`` ranks ``math_multiply`` first. Closing that gap
  properly needs embeddings over the tool descriptions (or bilingual
  descriptions). Until then :meth:`LexicalToolSelector.select` detects the
  all-zero case and offers **everything** rather than an arbitrary
  top-k -- a selector with no opinion must not act like it has one.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..observability import get_logger, log_event
from ..rag.retrieval import tokenize
from ..tools import ToolRegistry

logger = get_logger(__name__)


@dataclass(frozen=True)
class ToolSelection:
    """Which tools one think step is allowed to see, and why."""

    names: List[str]
    considered: int
    filtered: bool
    pinned: List[str] = field(default_factory=list)


class ToolSelector(Protocol):
    def select(
        self,
        registry: ToolRegistry,
        query: str,
        *,
        used: Sequence[str] = (),
    ) -> ToolSelection: ...


class AllToolsSelector:
    """Offers everything -- the dependency-free reference implementation, and
    exactly what a ``ReActLoop`` with no selector configured already does."""

    def select(
        self,
        registry: ToolRegistry,
        query: str,
        *,
        used: Sequence[str] = (),
    ) -> ToolSelection:
        names = registry.names()
        return ToolSelection(names=names, considered=len(names), filtered=False)


class LexicalToolSelector:
    """Picks the top-k tools whose name + description best match the query.

    Args:
        top_k: How many retrieved tools to offer, before pinned/used tools
            are added on top.
        min_tools: Registry size below which selection is skipped entirely.
        pinned: Tool names always offered regardless of score -- control-plane
            tools the task text will never lexically match.
    """

    def __init__(
        self,
        *,
        top_k: int = 8,
        min_tools: int = 12,
        pinned: Sequence[str] = (),
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        self.top_k = top_k
        self.min_tools = min_tools
        self.pinned = list(pinned)
        self.k1 = k1
        self.b = b

    def select(
        self,
        registry: ToolRegistry,
        query: str,
        *,
        used: Sequence[str] = (),
    ) -> ToolSelection:
        names = registry.names()
        if len(names) <= self.min_tools:
            return ToolSelection(names=names, considered=len(names), filtered=False)

        scores = self._score(registry, names, query)
        if not any(score > 0.0 for score in scores.values()):
            # Nothing matched at all, so the ranking is meaningless and
            # `sorted` would just hand back the first top_k in registry
            # order -- an arbitrary slice presented as a selection, which
            # is precisely the over-filtering this class is supposed to
            # guard against. No opinion means offer everything.
            #
            # The common cause is a language mismatch: this is lexical
            # matching, so a Chinese question against English tool
            # descriptions shares no terms and scores zero across the
            # board. Fixing *that* needs embeddings over tool descriptions
            # (or bilingual descriptions); until then this makes the miss
            # harmless instead of silently harmful. `tool_router.no_match`
            # in the logs is the signal that it is happening.
            log_event(
                logger,
                logging.INFO,
                "tool_router.select.no_match",
                considered=len(names),
                query_chars=len(query),
            )
            return ToolSelection(names=names, considered=len(names), filtered=False)

        # Only tools that actually matched are eligible: padding up to
        # top_k with zero-scoring tools would offer an arbitrary set under
        # the appearance of a ranking.
        matched = sorted(
            (item for item in scores.items() if item[1] > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked = [name for name, _score in matched][: self.top_k]

        always = [name for name in (*self.pinned, *used) if name in registry]
        # dict.fromkeys preserves registration order for the always-on set
        # while de-duplicating against the retrieved ones.
        selected = list(dict.fromkeys([*always, *ranked]))

        log_event(
            logger,
            logging.DEBUG,
            "tool_router.select.completed",
            considered=len(names),
            offered=len(selected),
            pinned=len(always),
            query_chars=len(query),
        )
        return ToolSelection(
            names=selected,
            considered=len(names),
            filtered=len(selected) < len(names),
            pinned=always,
        )

    def _score(
        self, registry: ToolRegistry, names: Sequence[str], query: str
    ) -> Dict[str, float]:
        """BM25 over the tool descriptions, treating each tool as one document."""

        documents: Dict[str, Counter] = {}
        for name in names:
            item = registry.get(name)
            text = f"{name} {getattr(item, 'description', '') or ''}"
            documents[name] = Counter(tokenize(text))

        total = max(1, len(documents))
        lengths = {name: sum(counts.values()) for name, counts in documents.items()}
        average = sum(lengths.values()) / total
        df: Counter = Counter()
        for counts in documents.values():
            df.update(counts.keys())

        terms = tokenize(query)
        scores: Dict[str, float] = {name: 0.0 for name in names}
        for name, counts in documents.items():
            length = lengths[name]
            for term in terms:
                frequency = counts.get(term, 0)
                if not frequency:
                    continue
                document_frequency = df.get(term, 0)
                idf = math.log(
                    1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
                )
                scores[name] += idf * frequency * (self.k1 + 1) / (
                    frequency + self.k1 * (1 - self.b + self.b * length / max(1.0, average))
                )
        return scores


def filtered_schemas(
    registry: ToolRegistry, selection: ToolSelection
) -> List[Dict[str, Any]]:
    """The schema list for ``selection``, in the registry's own order."""

    chosen = set(selection.names)
    return [
        item.to_schema()
        for name in registry.names()
        if name in chosen and (item := registry.get(name)) is not None
    ]
