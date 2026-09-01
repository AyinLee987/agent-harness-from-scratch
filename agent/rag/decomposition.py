"""Intent classification + sub-question decomposition for RAG queries.

``MedicalQueryPlanner.plan()`` already computes a ``subquestions`` field by
splitting on Chinese/English question marks and semicolons -- but nothing
downstream ever reads it (``RAGPipeline.retrieve()`` only ever issues one
retrieval, against the whole normalized text). This module replaces that
dead, purely-syntactic split with a real one: an LLM call that classifies a
question into one of three modes and, for the mode where it's actually
possible, produces sub-questions retrieval can act on independently.

The three modes, and why only one of them gets pre-split sub-questions:

* ``single_hop`` -- one retrieval answers it. No decomposition needed.
* ``parallel`` -- multiple *independent* sub-questions bundled into one
  message (e.g. "孕妇能不能吃布洛芬？另外成人推荐剂量是多少？"). Each
  sub-question's full text is already knowable from the original message,
  so they can all be retrieved separately and merged.
* ``sequential`` -- a later sub-question's content depends on an earlier
  retrieval's *result* (e.g. "治疗高血压的一线药物在老年人的推荐剂量是多
  少？" -- the drug name isn't in the question, it only exists after the
  first hop runs). This genuinely cannot be pre-split at classification
  time: the second sub-question's text doesn't exist yet. Rather than fake
  a split, this mode is left to the agent's own ``medical_evidence_search``
  tool calls (a normal ReAct loop can already chain them); the point of
  labeling it here is so the injected evidence context can tell the model
  to expect that instead of leaving it to guess.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from ..llm import BaseLLM

VALID_MODES = ("single_hop", "parallel", "sequential")


@dataclass
class QueryDecomposition:
    mode: str = "single_hop"
    subquestions: List[str] = field(default_factory=list)
    reasoning: str = ""


class QueryDecomposer(Protocol):
    def decompose(self, text: str) -> QueryDecomposition:
        ...


_SYSTEM_PROMPT = """你是一个检索问题分类器。给定用户的一句提问，判断它属于以下三类中的哪一类：

1. single_hop：一次检索就能回答，或者虽然复杂但是围绕同一个主体的单一问题。
2. parallel：包含多个互相独立、谁先谁后无所谓的子问题（每个子问题单独拿出来都是完整、可检索的）。
3. sequential：后一部分依赖前一部分检索出的具体结果才能提问（比如后半句提到的实体，
   其实是前半句在问"是什么"，本身还不知道具体是什么）——这种情况后续子问题的具体内容
   现在还不存在，不要编造。

只输出一个 JSON 对象，不要有任何其他文字：
{"mode": "single_hop" | "parallel" | "sequential", "subquestions": [...], "reasoning": "一句话说明判断依据"}

规则：
- mode 是 "parallel" 时，subquestions 必须是 2 个以上完整、独立、可直接拿去检索的问题
  （每个都要包含足够的上下文，不能是"另外呢"这种省略主语的片段）。
- mode 是 "single_hop" 或 "sequential" 时，subquestions 必须是空数组 []——sequential
  不预先编造依赖前一跳结果才能确定的子问题内容。
- 无法确定时，默认输出 single_hop。"""


class LLMQueryDecomposer:
    """Classifies a question and, for genuinely independent compound
    questions, splits it into sub-questions the pipeline can retrieve
    separately. Fail-closed: any parse/classification error falls back to
    ``single_hop`` with no sub-questions -- i.e. today's plain behavior --
    rather than guessing or raising.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def decompose(self, text: str) -> QueryDecomposition:
        try:
            response = self._llm.chat(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                tools=[],
            )
        except Exception:
            return QueryDecomposition()
        return _parse(response.content or "")


def _parse(raw: str) -> QueryDecomposition:
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return QueryDecomposition()
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return QueryDecomposition()
    mode = data.get("mode")
    if mode not in VALID_MODES:
        return QueryDecomposition()
    subquestions = data.get("subquestions") or []
    if not isinstance(subquestions, list):
        return QueryDecomposition()
    subquestions = [str(item).strip() for item in subquestions if str(item).strip()]
    if mode == "parallel" and len(subquestions) < 2:
        # A "parallel" verdict without real sub-questions isn't actionable --
        # fail closed rather than run a "parallel" retrieval of one item.
        return QueryDecomposition()
    if mode != "parallel":
        subquestions = []
    reasoning = str(data.get("reasoning") or "").strip()
    return QueryDecomposition(mode=mode, subquestions=subquestions, reasoning=reasoning)
