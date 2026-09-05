"""Fail-closed ingestion and version publishing for the RAG corpus."""

from __future__ import annotations

import threading
from typing import Any, Mapping, Optional, Protocol, Sequence

from .chunking import MedicalParentChildChunker, normalize_document_text
from .models import Chunk, Document, DocumentStatus, IngestionResult
from .repository import RAGRepository, content_checksum


class DocumentIndexer(Protocol):
    def index_document(self, document: Document, chunks: Sequence[Chunk]) -> None: ...


class RAGIngestionService:
    def __init__(
        self,
        repository: RAGRepository,
        chunker: MedicalParentChildChunker,
        indexers: Sequence[DocumentIndexer] = (),
    ) -> None:
        self.repository = repository
        self.chunker = chunker
        self.indexers = list(indexers)
        self._lock = threading.RLock()

    def ingest_text(
        self,
        *,
        logical_id: str,
        title: str,
        content: str,
        source_url: str = "",
        publisher: str = "unknown",
        document_type: str = "reference",
        jurisdiction: str = "",
        language: str = "zh-CN",
        version: str = "1",
        metadata: Optional[Mapping[str, Any]] = None,
        **dates: Any,
    ) -> IngestionResult:
        with self._lock:
            return self._ingest_text(
                logical_id=logical_id, title=title, content=content,
                source_url=source_url, publisher=publisher,
                document_type=document_type, jurisdiction=jurisdiction,
                language=language, version=version, metadata=metadata, **dates,
            )

    def _ingest_text(
        self,
        *,
        logical_id: str,
        title: str,
        content: str,
        source_url: str,
        publisher: str,
        document_type: str,
        jurisdiction: str,
        language: str,
        version: str,
        metadata: Optional[Mapping[str, Any]],
        **dates: Any,
    ) -> IngestionResult:
        normalized = normalize_document_text(content)
        if not normalized:
            raise ValueError("Cannot ingest an empty document.")
        checksum = content_checksum(normalized)
        duplicate = self.repository.find_by_checksum(checksum)
        # Only a re-ingest of what is *currently live* is a no-op. Matching
        # on content alone was wrong: publishing v1=alpha, then v2=beta,
        # then v3=alpha returned the superseded v1 with skipped=True and
        # left v2 active -- a legitimate rollback silently did nothing, and
        # the caller was handed a document that was not the effective one.
        # Version and effective dates are part of a document's identity in a
        # governed corpus, not just its bytes. See BUGS.md #20.
        if (
            duplicate
            and duplicate.logical_id == logical_id
            and duplicate.chunker_version == self.chunker.version
            and duplicate.status == DocumentStatus.ACTIVE
            and duplicate.version == version
        ):
            return IngestionResult(
                document=duplicate,
                chunks=self.repository.chunks_for_document(duplicate.id),
                skipped=True,
                warnings=["Identical normalized content is already indexed."],
            )
        reused_from = (
            duplicate
            if duplicate is not None and duplicate.logical_id == logical_id
            else None
        )
        document = Document(
            logical_id=logical_id,
            title=title,
            raw_content=content,
            normalized_content=normalized,
            checksum=checksum,
            source_url=source_url,
            publisher=publisher,
            document_type=document_type,
            jurisdiction=jurisdiction,
            language=language,
            version=version,
            chunker_version=self.chunker.version,
            metadata=dict(metadata or {}),
            **dates,
        )
        self.repository.insert_document(document)
        chunks: list[Chunk] = []
        try:
            chunks = self.chunker.chunk(document)
            validation = self.chunker.validate(document, chunks)
            if not validation.valid:
                raise ValueError("; ".join(validation.errors))
            self.repository.insert_chunks(chunks)
            for indexer in self.indexers:
                indexer.index_document(document, chunks)
            published = self.repository.publish(document.id)
            warnings = list(validation.warnings)
            if reused_from is not None:
                # Not an error -- a rollback or a re-publish under new
                # effective dates is a normal governance operation -- but
                # the operator should be able to see it happened.
                warnings.append(
                    f"Content is byte-identical to version {reused_from.version!r} "
                    f"({reused_from.status.value}); published as a new version "
                    f"{version!r} rather than skipped."
                )
            return IngestionResult(published, chunks, warnings=warnings)
        except Exception:
            failed = self.repository.get_document(document.id)
            if failed is not None and failed.status == DocumentStatus.STAGING:
                failed.status = DocumentStatus.FAILED
                self.repository.update_document(failed)
            raise
