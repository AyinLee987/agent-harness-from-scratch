"""Typed source, chunk, retrieval, and evidence models for RAG."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, Enum):
    STAGING = "staging"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    WITHDRAWN = "withdrawn"
    FAILED = "failed"


class EvidenceStatus(str, Enum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    STALE = "stale"
    RETRIEVAL_FAILED = "retrieval_failed"


@dataclass
class Document:
    logical_id: str
    title: str
    raw_content: str
    normalized_content: str
    checksum: str
    source_url: str = ""
    publisher: str = "unknown"
    document_type: str = "reference"
    jurisdiction: str = ""
    language: str = "zh-CN"
    version: str = "1"
    status: DocumentStatus = DocumentStatus.STAGING
    published_at: Optional[datetime] = None
    effective_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    supersedes_id: Optional[str] = None
    superseded_by_id: Optional[str] = None
    parser_version: str = "plain-structure-v1"
    chunker_version: str = "medical-parent-child-v1"
    ingested_at: datetime = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])


@dataclass
class Chunk:
    document_id: str
    text: str
    contextual_text: str
    section_path: List[str]
    sequence: int
    chunk_type: str = "child"
    parent_chunk_id: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    population: List[str] = field(default_factory=list)
    specialty: List[str] = field(default_factory=list)
    evidence_grade: Optional[str] = None
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])


@dataclass
class RetrievalFilters:
    jurisdiction: Optional[str] = None
    language: Optional[str] = None
    document_types: Optional[List[str]] = None
    publishers: Optional[List[str]] = None
    populations: Optional[List[str]] = None


@dataclass
class MedicalQuery:
    original: str
    normalized: str
    lexical_queries: List[str]
    semantic_queries: List[str]
    entities: List[str] = field(default_factory=list)
    filters: RetrievalFilters = field(default_factory=RetrievalFilters)
    subquestions: List[str] = field(default_factory=list)


@dataclass
class RetrievalHit:
    chunk_id: str
    score: float
    rank: int = 0
    retriever: str = ""


@dataclass
class Citation:
    document_id: str
    chunk_id: str
    title: str
    publisher: str
    version: str
    section_path: List[str]
    source_url: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None


@dataclass
class Evidence:
    document: Document
    chunk: Chunk
    parent_chunk: Optional[Chunk]
    citation: Citation
    bm25_score: Optional[float] = None
    dense_score: Optional[float] = None
    rrf_score: float = 0.0
    rerank_score: Optional[float] = None
    match_reason: str = "hybrid"


@dataclass
class EvidenceConflict:
    conflict_group: str
    evidence_ids: List[str]
    stances: List[str]


@dataclass
class EvidenceBundle:
    status: EvidenceStatus
    query: MedicalQuery
    evidence: List[Evidence] = field(default_factory=list)
    conflicts: List[EvidenceConflict] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    degraded_components: List[str] = field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return self.status == EvidenceStatus.SUFFICIENT


@dataclass
class IngestionResult:
    document: Document
    chunks: List[Chunk]
    skipped: bool = False
    warnings: List[str] = field(default_factory=list)
