"""Intent classification and query rewriting in front of the ReAct loop.

Without this, every request costs the same: a greeting and a clinical
question both start a full Leader run with every tool attached, and -- when
RAG is enabled -- both pay for a mandatory BM25 + dense retrieval before the
first token is generated (``RAGContextProvider`` runs unconditionally, see
``agent/rag/context.py``). The retrieval a greeting triggers is pure waste,
and the evidence block it injects is noise in the model's context.

This module adds one cheap call in front of that, producing a
:class:`RunPlan`: what kind of request this is, and the normalized text to
act on. Three routes, chosen because they map onto three genuinely
different costs:

* :attr:`Route.DIRECT` -- conversational or general-knowledge; the model
  answers from its own weights, one call, no tools, no retrieval.
* :attr:`Route.RETRIEVAL` -- a domain question the governed corpus is meant
  to answer; the mandatory evidence injection earns its cost here.
* :attr:`Route.REACT` -- needs tools, computation, or several steps; the
  full loop, with retrieval available as a *tool* rather than forced.

**Fail-open, and note this is the opposite of the RAG decomposer's
fail-closed default** (``agent/rag/decomposition.py``). There, a failed
classification falls back to retrieving *more*; here it falls back to
:attr:`Route.REACT`, the route with the most capability. Both rules are the
same rule underneath: when the classifier is unsure, never let it be the
thing that removes an ability the request might need.

The routing decision is also deliberately *recoverable* rather than final.
A ``direct`` answer that turns out to need a tool escalates back into the
full loop (see :data:`ESCALATION_SENTINEL`), so a misroute costs one extra
fast-model call instead of a wrong answer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Protocol

from ..llm import BaseLLM
from ..observability import get_logger, log_event

logger = get_logger(__name__)


class Route(str, Enum):
    """Where a request should start. A closed set, so it is an enum."""

    DIRECT = "direct"
    RETRIEVAL = "retrieval"
    REACT = "react"


#: What a ``direct`` answer must reply with instead of guessing when it
#: turns out to need a tool or the evidence corpus. Mirrors the
#: ``TASK_FAILED:`` convention ``examples/hierarchical_agent_kit.py``
#: already uses for a specialist that cannot finish its sub-task.
ESCALATION_SENTINEL = "NEEDS_TOOLS"


@dataclass
class RunPlan:
    """The routing decision for one request.

    Attributes:
        route: Which path to start on.
        task: The text downstream should act on -- the router's normalized
            rewrite, or the original when rewriting is disabled or failed.
        original_task: Exactly what the user sent, always preserved so a
            bad rewrite can be diagnosed (and logged) after the fact.
        needs_retrieval: Whether the mandatory evidence injection should
            run. True only for :attr:`Route.RETRIEVAL`; a ``react`` run
            still has the search *tool*, it just isn't forced to retrieve
            before it has decided it needs to.
        reasoning: The classifier's one-line justification, for logs.
    """

    route: Route
    task: str
    original_task: str
    needs_retrieval: bool
    reasoning: str = ""

    @property
    def rewritten(self) -> bool:
        return self.task.strip() != self.original_task.strip()


class QueryRouter(Protocol):
    def route(self, task: str) -> RunPlan: ...


class StaticRouter:
    """Always returns the same route -- the dependency-free reference
    implementation, and what tests use to pin a path without an LLM."""

    def __init__(self, route: Route = Route.REACT) -> None:
        self._route = route

    def route(self, task: str) -> RunPlan:
        return RunPlan(
            route=self._route,
            task=task,
            original_task=task,
            needs_retrieval=self._route is Route.RETRIEVAL,
            reasoning="static router",
        )


_SYSTEM_PROMPT = """你是一个请求分诊器。给定用户的一句话，做两件事：

(1) 归一化改写 normalized：去掉口语噪声和寒暄，补全省略的主语/指代，统一术语，
    保留全部信息量。不要回答问题，不要增加原文没有的限定条件。如果原文已经清楚，
    原样输出。

(2) 分类 route，三选一：
- direct：闲聊、寒暄、情绪表达，或者不依赖专业资料和工具、模型凭常识就能直接答的问题。
- retrieval：需要依据专业知识库/权威资料才能负责任地回答的问题（例如医疗、法律、
  产品规格这类"答错有代价、必须有出处"的领域问题）。
- react：需要调用工具才能完成的任务——计算、查日期、访问网页、读写文件、
  或者需要分成多步依次完成的复杂任务。

判断不了就输出 react。

只输出一个 JSON 对象，不要有任何其他文字：
{"route": "direct" | "retrieval" | "react", "normalized": "...", "reasoning": "一句话依据"}"""


class LLMQueryRouter:
    """Classifies and rewrites one request with a single fast-model call.

    Args:
        llm: The model to classify with -- intended to be the ``fast`` tier
            from ``config/agent.yaml``, since this is a short structured
            call whose output is cheap to sanity-check.
        domain_hint: One line describing what the retrieval corpus actually
            covers, appended to the prompt. Without it the classifier has
            to guess which domain questions are worth retrieving for, which
            is the single biggest source of ``retrieval``/``react``
            confusion in practice.
    """

    def __init__(self, llm: BaseLLM, *, domain_hint: str = "") -> None:
        self._llm = llm
        self._domain_hint = domain_hint.strip()

    def route(self, task: str) -> RunPlan:
        system = _SYSTEM_PROMPT
        if self._domain_hint:
            system = f"{system}\n\n知识库覆盖范围：{self._domain_hint}"
        try:
            response = self._llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": task},
                ],
                tools=[],
            )
        except Exception:
            # Fail open: a router outage must not take the agent down with
            # it, and must not quietly narrow what the run is allowed to do.
            log_event(logger, logging.WARNING, "router.route.failed", exc_info=True)
            return _fallback(task)

        plan = _parse(response.content or "", task)
        log_event(
            logger,
            logging.INFO,
            "router.route.decided",
            route=plan.route.value,
            needs_retrieval=plan.needs_retrieval,
            rewritten=plan.rewritten,
            task_chars=len(task),
        )
        return plan


def _fallback(task: str) -> RunPlan:
    return RunPlan(
        route=Route.REACT,
        task=task,
        original_task=task,
        needs_retrieval=False,
        reasoning="router unavailable; defaulted to the full loop",
    )


def _parse(raw: str, task: str) -> RunPlan:
    """Parse the classifier's JSON, falling back to :func:`_fallback`.

    A rewrite is only accepted if it is non-empty and not wildly shorter
    than the original: a model that "normalizes" a two-sentence question
    down to three characters has dropped the question, not cleaned it up.
    """

    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return _fallback(task)
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return _fallback(task)

    try:
        route = Route(str(data.get("route", "")).strip().lower())
    except ValueError:
        return _fallback(task)

    normalized = str(data.get("normalized") or "").strip()
    if not normalized or len(normalized) < max(2, len(task.strip()) // 4):
        normalized = task

    return RunPlan(
        route=route,
        task=normalized,
        original_task=task,
        needs_retrieval=route is Route.RETRIEVAL,
        reasoning=str(data.get("reasoning") or "").strip(),
    )


DIRECT_SYSTEM_PROMPT = (
    "You are a helpful assistant answering a short, conversational or "
    "general-knowledge question directly. You have no tools and no "
    "reference corpus on this path.\n\n"
    f"If answering well would actually require a tool (calculation, the "
    f"current date, fetching a page, reading a file) or authoritative "
    f"source material you do not have, reply with exactly "
    f"{ESCALATION_SENTINEL} and nothing else. Do not guess, and do not "
    f"apologize -- replying with {ESCALATION_SENTINEL} is the correct, "
    "expected outcome in that case and the request will be retried with "
    "full capability."
)


def wants_escalation(answer: Optional[str]) -> bool:
    """Whether a ``direct`` answer asked to be retried with full capability.

    Matched leniently (the sentinel appearing anywhere in a short reply)
    because a model that has been told to emit a bare token still tends to
    wrap it in punctuation or a sentence; the cost of a false positive is
    one wasted-but-correct full run, while a false negative ships a guess.
    """

    if not answer:
        return False
    return ESCALATION_SENTINEL in answer.upper()
