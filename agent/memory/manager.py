"""Policy-controlled durable memory coordinator."""

from __future__ import annotations

import hashlib
import copy
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set

from ..observability import get_logger, log_event
from .embeddings import EmbeddingProvider, provider_dimension
from .errors import MemoryNotFoundError, MemoryProtectedError
from .index import InMemoryVectorIndex, MemoryVectorIndex
from .models import (
    MemoryCandidate,
    MemoryDecision,
    MemoryKind,
    MemoryRecord,
    MemorySearchResult,
    MemoryStatus,
    RetentionPolicy,
    RunCompletedEvent,
    utc_now,
)
from .policy import DefaultMemoryPolicy, MemoryExtractor, MemoryPolicy, NoopMemoryExtractor
from .repository import InMemoryMemoryRepository, MemoryRepository

logger = get_logger(__name__)


class MemoryManager:
    """Coordinates extraction, policy, persistence, indexing, and lifecycle.

    ``repository`` is authoritative. ``vector_index`` is derived and can be
    rebuilt from active records at any time.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        repository: Optional[MemoryRepository] = None,
        vector_index: Optional[MemoryVectorIndex] = None,
        policy: Optional[MemoryPolicy] = None,
        extractor: Optional[MemoryExtractor] = None,
        ephemeral_ttl: timedelta = timedelta(hours=24),
    ) -> None:
        self.embedding_provider = embedding_provider
        self.repository = (
            repository if repository is not None else InMemoryMemoryRepository()
        )
        self.vector_index = (
            vector_index if vector_index is not None else InMemoryVectorIndex()
        )
        self.policy = policy or DefaultMemoryPolicy()
        self.extractor = extractor or NoopMemoryExtractor()
        self.ephemeral_ttl = ephemeral_ttl
        self._pending: Dict[str, MemoryCandidate] = {}
        self._lock = threading.RLock()

    def on_run_completed(self, event: RunCompletedEvent) -> List[MemoryRecord]:
        """Extract and process candidates; safe defaults produce no candidates."""

        if not event.success:
            log_event(
                logger,
                logging.INFO,
                "memory.run.skipped",
                run_id=event.run_id,
                reason="run_not_successful",
            )
            return []
        stored: List[MemoryRecord] = []
        for candidate in self.extractor.extract(event):
            candidate.source_run_id = candidate.source_run_id or event.run_id
            if candidate.namespace == "default":
                candidate.namespace = event.namespace
            if candidate.subject_id == "anonymous":
                candidate.subject_id = event.subject_id
            record = self.store_candidate(candidate)
            if record is not None:
                stored.append(record)
        return stored

    def store_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        confirmed: bool = False,
    ) -> Optional[MemoryRecord]:
        """Apply policy and persist, queue for confirmation, or skip a candidate."""

        decision = MemoryDecision.PERSIST if confirmed else self.policy.decide(candidate)
        if decision == MemoryDecision.SKIP:
            log_event(
                logger,
                logging.INFO,
                "memory.write.skipped",
                candidate_id=candidate.id,
                kind=candidate.kind.value,
                reason="policy",
            )
            return None
        if decision == MemoryDecision.REQUIRE_CONFIRMATION:
            with self._lock:
                self._pending[candidate.id] = candidate
            log_event(
                logger,
                logging.INFO,
                "memory.write.pending_confirmation",
                candidate_id=candidate.id,
                kind=candidate.kind.value,
                sensitivity=candidate.sensitivity.value,
            )
            return None
        if decision == MemoryDecision.EPHEMERAL and candidate.expires_at is None:
            candidate.expires_at = utc_now() + self.ephemeral_ttl
        return self._persist(candidate)

    def pending_candidates(self) -> List[MemoryCandidate]:
        with self._lock:
            return copy.deepcopy(list(self._pending.values()))

    def confirm(self, candidate_id: str) -> MemoryRecord:
        with self._lock:
            candidate = self._pending.pop(candidate_id, None)
        if candidate is None:
            raise MemoryNotFoundError(f"Pending memory candidate {candidate_id!r} not found.")
        candidate.metadata = {**candidate.metadata, "confirmed": True}
        record = self._persist(candidate)
        log_event(
            logger,
            logging.INFO,
            "memory.write.confirmed",
            candidate_id=candidate_id,
            record_id=record.id,
        )
        return record

    def reject(self, candidate_id: str) -> bool:
        with self._lock:
            rejected = self._pending.pop(candidate_id, None) is not None
        if rejected:
            log_event(
                logger,
                logging.INFO,
                "memory.write.rejected",
                candidate_id=candidate_id,
            )
        return rejected

    def recall(
        self,
        query: str,
        *,
        namespace: str = "default",
        subject_id: str = "anonymous",
        kinds: Optional[Set[MemoryKind]] = None,
        limit: int = 10,
        min_score: Optional[float] = None,
    ) -> List[MemorySearchResult]:
        if not query.strip() or limit <= 0:
            return []
        vector = self.embedding_provider.embed_query(query)
        provider_dimension(self.embedding_provider, vector)
        hits = self.vector_index.search(
            vector,
            limit=limit,
            namespace=namespace,
            subject_id=subject_id,
            kinds=kinds,
        )
        results: List[MemorySearchResult] = []
        for hit in hits:
            if min_score is not None and hit.score < min_score:
                continue
            record = self.repository.get(hit.record_id)
            if record is None or record.status != MemoryStatus.ACTIVE:
                continue
            record.last_accessed_at = utc_now()
            record.updated_at = record.last_accessed_at
            self.repository.update(record)
            results.append(
                MemorySearchResult(record=record, semantic_score=hit.score)
            )
        log_event(
            logger,
            logging.INFO,
            "memory.read.completed",
            namespace=namespace,
            subject_hash=hashlib.sha256(subject_id.encode("utf-8")).hexdigest()[:12],
            result_count=len(results),
            query_chars=len(query),
        )
        return results

    def as_search_tool(
        self,
        *,
        namespace: str,
        subject_id: str,
        limit: int = 5,
    ):
        """Expose scoped recall as a read-only tool; writes remain policy-controlled."""

        memory = self

        def _search(query: str) -> str:
            """Search approved memories for the current user and namespace.

            Args:
                query: Natural-language description of what should be recalled.
            """

            results = memory.recall(
                query,
                namespace=namespace,
                subject_id=subject_id,
                limit=limit,
            )
            if not results:
                return "No approved memories found."
            return "\n".join(
                f"- [{item.record.id}] ({item.record.kind.value}) "
                f"{item.record.content} (relevance: {item.final_score:.2f})"
                for item in results
            )

        from ..tools import FunctionTool

        return FunctionTool(_search, name="memory_search")

    def supersede(
        self,
        old_record_id: str,
        replacement: MemoryCandidate,
        *,
        confirmed: bool = False,
    ) -> MemoryRecord:
        """Insert a new version and remove the old version from active retrieval."""

        with self._lock:
            old = self._require_record(old_record_id)
            if old.status != MemoryStatus.ACTIVE:
                raise MemoryProtectedError("Only active memory can be superseded.")
            decision = MemoryDecision.PERSIST if confirmed else self.policy.decide(replacement)
            if decision not in {MemoryDecision.PERSIST, MemoryDecision.EPHEMERAL}:
                raise MemoryProtectedError(
                    f"Replacement memory was not approved for persistence: {decision.value}."
                )
            replacement.namespace = old.namespace
            replacement.subject_id = old.subject_id
            new = self._persist(
                replacement,
                supersedes_id=old.id,
                version=old.version + 1,
                allow_duplicate=False,
            )
            try:
                old.status = MemoryStatus.SUPERSEDED
                old.updated_at = utc_now()
                self.repository.update(old)
            except Exception:
                # Keep the old record retrievable and quarantine the incomplete
                # replacement so a partial version transition is never active.
                new.status = MemoryStatus.QUARANTINED
                new.metadata = {**new.metadata, "supersede_incomplete": True}
                new.updated_at = utc_now()
                self.repository.update(new)
                self.vector_index.delete(new.id)
                raise
            self.vector_index.delete(old.id)
            log_event(
                logger,
                logging.INFO,
                "memory.superseded",
                old_record_id=old.id,
                new_record_id=new.id,
            )
            return new

    def expire_due(self, now: Optional[datetime] = None) -> List[str]:
        """Expire only TTL/ephemeral records; protected records are never auto-deleted."""

        instant = now or utc_now()
        expired: List[str] = []
        with self._lock:
            for record in self.repository.list_records(status=MemoryStatus.ACTIVE):
                if record.pinned:
                    continue
                if record.retention_policy not in {
                    RetentionPolicy.EPHEMERAL,
                    RetentionPolicy.TTL,
                }:
                    continue
                if record.expires_at is None or record.expires_at > instant:
                    continue
                record.status = MemoryStatus.EXPIRED
                record.updated_at = instant
                self.repository.update(record)
                self.vector_index.delete(record.id)
                expired.append(record.id)
        if expired:
            log_event(
                logger,
                logging.INFO,
                "memory.expired",
                record_count=len(expired),
            )
        return expired

    def records_due_for_review(self, now: Optional[datetime] = None) -> List[MemoryRecord]:
        instant = now or utc_now()
        return [
            record
            for record in self.repository.list_records(status=MemoryStatus.ACTIVE)
            if record.review_at is not None and record.review_at <= instant
        ]

    def quarantine(self, record_id: str, reason: str) -> MemoryRecord:
        with self._lock:
            record = self._require_record(record_id)
            record.status = MemoryStatus.QUARANTINED
            record.updated_at = utc_now()
            record.metadata = {**record.metadata, "quarantine_reason": reason}
            self.repository.update(record)
            self.vector_index.delete(record.id)
            log_event(
                logger,
                logging.WARNING,
                "memory.quarantined",
                record_id=record.id,
            )
            return record

    def tombstone(
        self,
        record_id: str,
        *,
        reason: str,
        allow_protected: bool = False,
    ) -> MemoryRecord:
        """Logically delete a record and immediately remove it from retrieval."""

        with self._lock:
            record = self._require_record(record_id)
            if record.pinned and not allow_protected:
                raise MemoryProtectedError(
                    "Pinned memory requires an explicitly authorized deletion."
                )
            record.status = MemoryStatus.TOMBSTONED
            record.deletion_reason = reason
            record.deleted_at = utc_now()
            record.updated_at = record.deleted_at
            self.repository.update(record)
            self.vector_index.delete(record.id)
            log_event(
                logger,
                logging.INFO,
                "memory.tombstoned",
                record_id=record.id,
                protected=record.pinned,
            )
            return record

    def purge_tombstones(self, deleted_before: datetime) -> List[str]:
        """Physically remove tombstones after the caller's configured grace period."""

        purged: List[str] = []
        with self._lock:
            records = self.repository.list_records(status=MemoryStatus.TOMBSTONED)
            for record in records:
                if record.deleted_at is None or record.deleted_at > deleted_before:
                    continue
                if self.repository.delete_physical(record.id):
                    purged.append(record.id)
        if purged:
            log_event(
                logger,
                logging.INFO,
                "memory.purged",
                record_count=len(purged),
            )
        return purged

    def rebuild_index(self) -> int:
        """Re-embed active source records into the derived vector index."""

        count = 0
        for record in self.repository.list_records(status=MemoryStatus.ACTIVE):
            vector = self.embedding_provider.embed_documents([record.content])[0]
            dimension = provider_dimension(self.embedding_provider, vector)
            if (
                record.embedding_model
                and record.embedding_model != self.embedding_provider.model_id
            ):
                record.metadata = {
                    **record.metadata,
                    "previous_embedding_model": record.embedding_model,
                }
            record.embedding_model = self.embedding_provider.model_id
            record.embedding_dimension = dimension
            record.updated_at = utc_now()
            self.repository.update(record)
            self.vector_index.upsert(
                record.id,
                vector,
                namespace=record.namespace,
                subject_id=record.subject_id,
                kind=record.kind,
            )
            count += 1
        return count

    def _persist(
        self,
        candidate: MemoryCandidate,
        *,
        supersedes_id: Optional[str] = None,
        version: int = 1,
        allow_duplicate: bool = True,
    ) -> MemoryRecord:
        if not candidate.content.strip():
            raise ValueError("Memory content must be non-empty.")
        with self._lock:
            if allow_duplicate:
                existing = self.repository.find_active_by_hash(candidate.content_hash())
                if existing is not None:
                    return existing

            vector = self.embedding_provider.embed_documents([candidate.content])[0]
            dimension = provider_dimension(self.embedding_provider, vector)
            record = MemoryRecord.from_candidate(
                candidate,
                embedding_model=self.embedding_provider.model_id,
                embedding_dimension=dimension,
                supersedes_id=supersedes_id,
                version=version,
            )
            self.repository.insert(record)
            try:
                self.vector_index.upsert(
                    record.id,
                    vector,
                    namespace=record.namespace,
                    subject_id=record.subject_id,
                    kind=record.kind,
                )
            except Exception:
                record.status = MemoryStatus.QUARANTINED
                record.metadata = {**record.metadata, "index_error": True}
                record.updated_at = utc_now()
                self.repository.update(record)
                raise
            log_event(
                logger,
                logging.INFO,
                "memory.write.completed",
                record_id=record.id,
                kind=record.kind.value,
                retention_policy=record.retention_policy.value,
                content_chars=len(record.content),
            )
            return record

    def _require_record(self, record_id: str) -> MemoryRecord:
        record = self.repository.get(record_id)
        if record is None:
            raise MemoryNotFoundError(f"Memory record {record_id!r} not found.")
        return record
