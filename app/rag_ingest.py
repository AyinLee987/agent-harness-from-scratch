"""Standalone command-line entry point for governed RAG corpus ingestion.

Run as ``python -m app.rag_ingest ...`` from the repo root (see README).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# This module lives at <repo root>/app/rag_ingest.py — one level below the
# repo root, where the `agent` package lives. `python -m app.rag_ingest`
# already puts the repo root on sys.path; this also makes a direct
# `python app/rag_ingest.py` invocation work.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

from agent import (
    BM25Retriever, DenseRetriever, MedicalParentChildChunker,
    OpenAICompatibleEmbeddingProvider, RAGIngestionService, SQLiteRAGRepository,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Add a file or directory to the RAG corpus.")
    result.add_argument("path", type=Path, help="UTF-8 .md/.txt file or directory")
    result.add_argument("--logical-id", help="Stable id; required only to override a single file id")
    result.add_argument("--title", help="Document title; defaults to the filename")
    result.add_argument("--publisher", default=os.getenv("RAG_DEFAULT_PUBLISHER", "unknown"))
    result.add_argument("--document-type", default="reference")
    result.add_argument("--jurisdiction", default=os.getenv("RAG_DEFAULT_JURISDICTION", "CN"))
    result.add_argument("--language", default="zh-CN")
    result.add_argument("--version", default="1")
    result.add_argument("--source-url", default="")
    result.add_argument("--metadata-json", default="{}", help="JSON object attached to each document")
    result.add_argument("--db", type=Path, default=Path(os.getenv("RAG_DB_PATH", "data/rag.sqlite")))
    return result


def main() -> int:
    load_dotenv()
    args = parser().parse_args()
    source = args.path.resolve()
    if not source.exists():
        raise SystemExit(f"Input does not exist: {source}")
    if source.is_dir() and (args.logical_id or args.title or args.source_url):
        raise SystemExit("--logical-id, --title, and --source-url are only valid for a single file.")
    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--metadata-json is invalid: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("--metadata-json must contain a JSON object.")
    model = os.getenv("RAG_EMBEDDING_MODEL") or os.getenv("OPENAI_EMBED_MODEL")
    if not model:
        raise SystemExit("Set RAG_EMBEDDING_MODEL before ingestion.")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    repository = SQLiteRAGRepository(args.db)
    try:
        embeddings = OpenAICompatibleEmbeddingProvider(
            model=model,
            api_key=os.getenv("RAG_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("RAG_EMBEDDING_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            provider_name="rag",
        )
        bm25 = BM25Retriever(repository)
        dense = DenseRetriever(repository, embeddings)
        ingestion = RAGIngestionService(
            repository, MedicalParentChildChunker(), [bm25, dense]
        )
        files = [source] if source.is_file() else sorted(
            item for item in source.rglob("*") if item.is_file() and item.suffix.lower() in {".md", ".txt"}
        )
        if not files:
            raise SystemExit("No .md or .txt files were found.")
        failed = 0
        for item in files:
            logical_id = args.logical_id or (
                item.name if source.is_file() else item.relative_to(source).as_posix()
            )
            try:
                result = ingestion.ingest_text(
                    logical_id=logical_id,
                    title=args.title or item.stem,
                    content=item.read_text(encoding="utf-8"),
                    source_url=args.source_url or item.as_uri(),
                    publisher=args.publisher,
                    document_type=args.document_type,
                    jurisdiction=args.jurisdiction,
                    language=args.language,
                    version=args.version,
                    metadata=metadata,
                )
                action = "skipped" if result.skipped else "published"
                print(f"{action}: {item} -> {result.document.id} ({len(result.chunks)} chunks)")
            except Exception as exc:
                failed += 1
                print(f"failed: {item}: {exc}")
        return 1 if failed else 0
    finally:
        repository.close()


if __name__ == "__main__":
    raise SystemExit(main())
