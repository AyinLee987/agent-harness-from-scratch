"""Hybrid retrieval pipeline that returns traceable, gated evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional

from .decomposition import QueryDecomposer, QueryDecomposition
from .models import (
    Citation, Evidence, EvidenceBundle, EvidenceConflict, EvidenceStatus,
    MedicalQuery, RetrievalFilters, utc_now,
)
from .query import MedicalQueryPlanner
from .rerank import HeuristicReranker, Reranker
from .repository import RAGRepository
from .retrieval import Retriever, reciprocal_rank_fusion

_SEQUENTIAL_HINT = (
    "这个问题看起来需要分步检索：下面这次检索只覆盖了整句话的字面意思，可能查不全。"
    "如果答案依赖某个具体实体（比如药名、病名）而这个实体还没在证据里出现，"
    "先从已有证据或你的推理里确定这个实体，再调用 medical_evidence_search 用这个"
    "具体实体重新查一次，不要在证据不全的情况下直接编造。"
)


@dataclass
class RAGConfig:
    candidate_limit: int = 30
    evidence_limit: int = 8
    minimum_evidence: int = 1
    minimum_rerank_score: float = 0.0
    require_multiple_documents: bool = False
    max_evidence_age_days: Optional[int] = None


class RAGPipeline:
    def __init__(
        self,
        repository: RAGRepository,
        lexical: Optional[Retriever],
        dense: Optional[Retriever],
        *,
        query_planner: Optional[MedicalQueryPlanner] = None,
        reranker: Optional[Reranker] = None,
        config: Optional[RAGConfig] = None,
        decomposer: Optional[QueryDecomposer] = None,
    ) -> None:
        if lexical is None and dense is None:
            raise ValueError("At least one retriever is required.")
        self.repository = repository
        self.lexical = lexical
        self.dense = dense
        self.query_planner = query_planner or MedicalQueryPlanner()
        self.reranker = reranker or HeuristicReranker()
        self.config = config or RAGConfig()
        self.decomposer = decomposer

    def retrieve(self, text: str, filters: RetrievalFilters | None = None) -> EvidenceBundle:
        """Retrieve evidence for ``text``.

        With no ``decomposer`` configured, behaves exactly as before: one
        retrieval against the whole text. With one configured, the question
        is first classified (single_hop / parallel / sequential -- see
        ``agent/rag/decomposition.py``); a "parallel" verdict fans out into
        one retrieval per independent sub-question and merges the results,
        a "sequential" verdict runs the normal single retrieval but attaches
        a hint telling the model to chain ``medical_evidence_search`` calls,
        and any classification failure fails closed to plain single_hop.
        """
        decomposition: Optional[QueryDecomposition] = None
        if self.decomposer is not None:
            try:
                decomposition = self.decomposer.decompose(text)
            except Exception:
                decomposition = None
        if decomposition is not None and decomposition.mode == "parallel" and decomposition.subquestions:
            return self._retrieve_parallel(text, decomposition, filters)
        bundle = self._retrieve_single(text, filters)
        if decomposition is not None and decomposition.mode == "sequential":
            bundle.query.mode = "sequential"
            bundle.decomposition_hint = _SEQUENTIAL_HINT
        return bundle

    def _retrieve_parallel(
        self, original_text: str, decomposition: QueryDecomposition, filters: RetrievalFilters | None,
    ) -> EvidenceBundle:
        sub_bundles = [self._retrieve_single(sq, filters) for sq in decomposition.subquestions]
        query = self.query_planner.plan(original_text, filters)
        query.subquestions = list(decomposition.subquestions)
        query.mode = "parallel"
        return _merge_parallel_bundles(query, sub_bundles)

    def _retrieve_single(self, text: str, filters: RetrievalFilters | None) -> EvidenceBundle:
        query = self.query_planner.plan(text, filters)
        result_sets: Dict[str, list] = {}
        degraded: List[str] = []
        for retriever in (self.lexical, self.dense):
            if retriever is None:
                continue
            try:
                result_sets[retriever.name] = retriever.search(query, self.config.candidate_limit)
            except Exception:
                degraded.append(retriever.name)
        if not result_sets:
            return EvidenceBundle(
                EvidenceStatus.RETRIEVAL_FAILED, query,
                missing_information=["All configured retrievers failed."],
                degraded_components=degraded,
            )
        source_scores = {
            name: {hit.chunk_id: hit.score for hit in hits} for name, hits in result_sets.items()
        }
        evidence: List[Evidence] = []
        seen_hashes: set[str] = set()
        for fused in reciprocal_rank_fusion(result_sets):
            chunk = self.repository.get_chunk(fused.chunk_id)
            document = self.repository.get_document(chunk.document_id) if chunk else None
            if not chunk or not document or chunk.content_hash in seen_hashes:
                continue
            seen_hashes.add(chunk.content_hash)
            parent = self.repository.get_chunk(chunk.parent_chunk_id) if chunk.parent_chunk_id else None
            evidence.append(Evidence(
                document=document,
                chunk=chunk,
                parent_chunk=parent,
                citation=Citation(
                    document.id, chunk.id, document.title, document.publisher,
                    document.version, chunk.section_path, document.source_url,
                    chunk.page_start, chunk.page_end,
                ),
                bm25_score=source_scores.get("bm25", {}).get(chunk.id),
                dense_score=source_scores.get("dense", {}).get(chunk.id),
                rrf_score=fused.score,
            ))
        try:
            evidence = self.reranker.rerank(query, evidence)
        except Exception:
            degraded.append("reranker")
        evidence = [
            item for item in evidence
            if (item.rerank_score if item.rerank_score is not None else item.rrf_score)
            >= self.config.minimum_rerank_score
        ][:self.config.evidence_limit]
        conflicts = self._find_conflicts(evidence)
        documents = {item.document.id for item in evidence}
        enough = len(evidence) >= self.config.minimum_evidence and (
            not self.config.require_multiple_documents or len(documents) >= 2
        )
        stale = self._all_stale(evidence)
        if conflicts:
            status = EvidenceStatus.CONFLICTING
        elif stale:
            status = EvidenceStatus.STALE
        elif enough:
            status = EvidenceStatus.SUFFICIENT
        else:
            status = EvidenceStatus.INSUFFICIENT
        if stale:
            missing = ["Matching evidence exists but is older than the configured review window."]
        else:
            missing = [] if enough else ["Not enough independently matching evidence was retrieved."]
        return EvidenceBundle(status, query, evidence, conflicts, missing, degraded)

    def _all_stale(self, evidence: List[Evidence]) -> bool:
        days = self.config.max_evidence_age_days
        if days is None or not evidence:
            return False
        cutoff = utc_now() - timedelta(days=days)
        return all(
            (item.document.reviewed_at or item.document.effective_at or item.document.published_at)
            is not None
            and (item.document.reviewed_at or item.document.effective_at or item.document.published_at) < cutoff
            for item in evidence
        )

    @staticmethod
    def _find_conflicts(evidence: List[Evidence]) -> List[EvidenceConflict]:
        groups: Dict[str, Dict[str, List[str]]] = {}
        for item in evidence:
            group = str(item.chunk.metadata.get("conflict_group") or item.document.metadata.get("conflict_group", ""))
            stance = str(item.chunk.metadata.get("stance") or item.document.metadata.get("stance", ""))
            if group and stance:
                groups.setdefault(group, {}).setdefault(stance, []).append(item.chunk.id)
        return [
            EvidenceConflict(group, [item for ids in stances.values() for item in ids], list(stances))
            for group, stances in groups.items() if len(stances) > 1
        ]


def _merge_parallel_bundles(query: MedicalQuery, sub_bundles: List[EvidenceBundle]) -> EvidenceBundle:
    """Combine one EvidenceBundle per independent sub-question into one.

    Conservative by design (matches the fail-closed spirit of the rest of
    this module): the merged bundle is only SUFFICIENT if *every*
    sub-question individually got sufficient evidence -- a parallel
    compound question half-answered is not "sufficient," it's silently
    dropping half the user's question.
    """
    evidence: List[Evidence] = []
    seen_hashes: set[str] = set()
    conflicts: List[EvidenceConflict] = []
    degraded: List[str] = []
    missing: List[str] = []
    for sub in sub_bundles:
        for item in sub.evidence:
            if item.chunk.content_hash in seen_hashes:
                continue
            seen_hashes.add(item.chunk.content_hash)
            evidence.append(item)
        conflicts.extend(sub.conflicts)
        for name in sub.degraded_components:
            if name not in degraded:
                degraded.append(name)
        if sub.status != EvidenceStatus.SUFFICIENT:
            detail = " ".join(sub.missing_information) or sub.status.value
            missing.append(f"子问题「{sub.query.normalized}」: {detail}")

    statuses = {sub.status for sub in sub_bundles}
    if conflicts:
        status = EvidenceStatus.CONFLICTING
    elif statuses == {EvidenceStatus.RETRIEVAL_FAILED}:
        status = EvidenceStatus.RETRIEVAL_FAILED
    elif statuses <= {EvidenceStatus.SUFFICIENT, EvidenceStatus.STALE} and EvidenceStatus.STALE in statuses:
        status = EvidenceStatus.STALE
    elif statuses == {EvidenceStatus.SUFFICIENT}:
        status = EvidenceStatus.SUFFICIENT
    else:
        status = EvidenceStatus.INSUFFICIENT
    return EvidenceBundle(status, query, evidence, conflicts, missing, degraded)


class CitationCounter:
    """Shared, mutable [E#] numbering state across multiple
    ``format_evidence_context`` calls within one run.

    Without one, every call restarts at [E1] -- fine when a run only ever
    retrieves once, but the mandatory ``RAGContextProvider`` injection and
    each ``medical_evidence_search`` follow-up call each call
    ``format_evidence_context`` independently, so a run with a follow-up
    search ends up with two *different* pieces of evidence both labeled
    [E1] -- a citation collision a model citing across both passes can't be
    told apart (found empirically running examples/rag_multihop_eval.py;
    see README's "Query decomposition" section).

    Pass the *same* instance to both a ``RAGContextProvider`` and a
    ``medical_evidence_search`` tool built from ``create_rag_search_tool``
    (as ``app/server.py`` does) so their citations share one numbering
    sequence for the run. Not thread-safe by design -- tool calls within
    one ReAct run are dispatched sequentially (see
    ``agent/trigger/react_loop.py``'s ``_act_node``), so there's nothing to
    guard against; don't share one instance across concurrent runs.
    """

    def __init__(self, start: int = 1) -> None:
        self._next = start

    def take(self, count: int) -> int:
        """Reserve ``count`` consecutive indices; return the first one."""
        first = self._next
        self._next += count
        return first


def format_evidence_context(
    bundle: EvidenceBundle, citation_counter: Optional[CitationCounter] = None,
) -> str:
    """Render ``bundle`` as the text injected into the model's context.

    ``citation_counter`` is optional and defaults to a fresh, private one
    (today's behavior: this call's citations start at [E1]). Pass a shared
    ``CitationCounter`` to make citations from multiple calls in the same
    run number continuously instead of colliding -- see ``CitationCounter``.
    """
    counter = citation_counter or CitationCounter()
    start_index = counter.take(len(bundle.evidence))
    lines = [f"证据状态: {bundle.status.value}", f"检索问题: {bundle.query.normalized}"]
    if bundle.query.mode != "single_hop":
        lines.append(f"问题类型: {bundle.query.mode}")
    if bundle.query.subquestions:
        lines.append("拆解出的子问题: " + " | ".join(bundle.query.subquestions))
    if bundle.decomposition_hint:
        lines.append("多跳提示: " + bundle.decomposition_hint)
    if bundle.degraded_components:
        lines.append("降级组件: " + ", ".join(bundle.degraded_components))
    if bundle.missing_information:
        lines.append("缺失信息: " + " ".join(bundle.missing_information))
    for offset, item in enumerate(bundle.evidence):
        index = start_index + offset
        citation = item.citation
        lines.extend([
            f"[E{index}] {citation.title} | {citation.publisher} | v{citation.version} | "
            f"{' > '.join(citation.section_path)} | document_id={citation.document_id} | chunk_id={citation.chunk_id}",
            item.chunk.text,
            f"来源: {citation.source_url or '未提供'}",
        ])
    if bundle.conflicts:
        lines.append("检测到冲突证据，必须向用户明确说明，不能自行消解。")
    return "\n".join(lines)
