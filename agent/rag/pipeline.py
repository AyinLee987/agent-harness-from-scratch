"""Hybrid retrieval pipeline that returns traceable, gated evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Optional

from .models import (
    Citation, Evidence, EvidenceBundle, EvidenceConflict, EvidenceStatus,
    MedicalQuery, RetrievalFilters, utc_now,
)
from .query import MedicalQueryPlanner
from .rerank import HeuristicReranker, Reranker
from .repository import RAGRepository
from .retrieval import Retriever, reciprocal_rank_fusion


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
    ) -> None:
        if lexical is None and dense is None:
            raise ValueError("At least one retriever is required.")
        self.repository = repository
        self.lexical = lexical
        self.dense = dense
        self.query_planner = query_planner or MedicalQueryPlanner()
        self.reranker = reranker or HeuristicReranker()
        self.config = config or RAGConfig()

    def retrieve(self, text: str, filters: RetrievalFilters | None = None) -> EvidenceBundle:
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


def format_evidence_context(bundle: EvidenceBundle) -> str:
    lines = [f"证据状态: {bundle.status.value}", f"检索问题: {bundle.query.normalized}"]
    if bundle.degraded_components:
        lines.append("降级组件: " + ", ".join(bundle.degraded_components))
    if bundle.missing_information:
        lines.append("缺失信息: " + " ".join(bundle.missing_information))
    for index, item in enumerate(bundle.evidence, 1):
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
