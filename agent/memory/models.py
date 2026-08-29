"""Typed records and lifecycle states for durable agent memory."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MemoryKind(str, Enum):
    SESSION = "session"
    USER_PREFERENCE = "user_preference"
    USER_FACT = "user_fact"
    EPISODE = "episode"
    TASK_OUTCOME = "task_outcome"
    AGENT_LEARNING = "agent_learning"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"
    TOMBSTONED = "tombstoned"
    QUARANTINED = "quarantined"


class RetentionPolicy(str, Enum):
    EPHEMERAL = "ephemeral"
    TTL = "ttl"
    UNTIL_REVIEW = "until_review"
    UNTIL_SUPERSEDED = "until_superseded"
    PINNED = "pinned"
    EXPLICIT_DELETE_ONLY = "explicit_delete_only"


class MemoryDecision(str, Enum):
    SKIP = "skip"
    EPHEMERAL = "ephemeral"
    PERSIST = "persist"
    REQUIRE_CONFIRMATION = "require_confirmation"


class Sensitivity(str, Enum):
    NORMAL = "normal"
    PERSONAL = "personal"
    HEALTH = "health"
    SECRET = "secret"


@dataclass
class MemoryCandidate:
    content: str
    kind: MemoryKind
    namespace: str = "default"
    subject_id: str = "anonymous"
    source_type: str = "unknown"
    source_ref: str = ""
    source_run_id: Optional[str] = None
    confidence: float = 1.0
    importance: float = 0.5
    sensitivity: Sensitivity = Sensitivity.NORMAL
    verification_status: str = "unverified"
    retention_policy: RetentionPolicy = RetentionPolicy.EXPLICIT_DELETE_ONLY
    expires_at: Optional[datetime] = None
    review_at: Optional[datetime] = None
    pinned: bool = False
    explicit_user_request: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def content_hash(self) -> str:
        normalized = " ".join(self.content.casefold().split())
        identity = f"{self.namespace}\0{self.subject_id}\0{self.kind.value}\0{normalized}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()


@dataclass
class MemoryRecord:
    id: str
    namespace: str
    subject_id: str
    kind: MemoryKind
    content: str
    source_type: str
    source_ref: str = ""
    source_run_id: Optional[str] = None
    confidence: float = 1.0
    importance: float = 0.5
    sensitivity: Sensitivity = Sensitivity.NORMAL
    verification_status: str = "unverified"
    status: MemoryStatus = MemoryStatus.ACTIVE
    version: int = 1
    supersedes_id: Optional[str] = None
    retention_policy: RetentionPolicy = RetentionPolicy.EXPLICIT_DELETE_ONLY
    expires_at: Optional[datetime] = None
    review_at: Optional[datetime] = None
    pinned: bool = False
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    last_accessed_at: Optional[datetime] = None
    embedding_model: Optional[str] = None
    embedding_dimension: Optional[int] = None
    content_hash: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    deletion_reason: Optional[str] = None
    deleted_at: Optional[datetime] = None

    @classmethod
    def from_candidate(
        cls,
        candidate: MemoryCandidate,
        *,
        embedding_model: Optional[str] = None,
        embedding_dimension: Optional[int] = None,
        supersedes_id: Optional[str] = None,
        version: int = 1,
    ) -> "MemoryRecord":
        return cls(
            id=uuid.uuid4().hex[:12],
            namespace=candidate.namespace,
            subject_id=candidate.subject_id,
            kind=candidate.kind,
            content=candidate.content.strip(),
            source_type=candidate.source_type,
            source_ref=candidate.source_ref,
            source_run_id=candidate.source_run_id,
            confidence=max(0.0, min(1.0, candidate.confidence)),
            importance=max(0.0, min(1.0, candidate.importance)),
            sensitivity=candidate.sensitivity,
            verification_status=candidate.verification_status,
            version=version,
            supersedes_id=supersedes_id,
            retention_policy=candidate.retention_policy,
            expires_at=candidate.expires_at,
            review_at=candidate.review_at,
            pinned=candidate.pinned
            or candidate.retention_policy == RetentionPolicy.PINNED,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
            content_hash=candidate.content_hash(),
            metadata=dict(candidate.metadata),
        )


@dataclass
class MemorySearchResult:
    record: MemoryRecord
    semantic_score: float
    lexical_score: float = 0.0
    final_score: float = 0.0
    match_reason: str = "semantic"

    def __post_init__(self) -> None:
        if self.final_score == 0.0:
            self.final_score = self.semantic_score


@dataclass
class RunCompletedEvent:
    run_id: str
    task: str
    answer: str
    success: bool
    stop_reason: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    namespace: str = "default"
    subject_id: str = "anonymous"


@dataclass
class SummarySnapshot:
    conversation_id: str
    summary: str
    through_message_id: Optional[str] = None
    updated_at: datetime = field(default_factory=utc_now)
