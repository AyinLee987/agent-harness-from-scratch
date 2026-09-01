"""Governed, hybrid RAG subsystem."""

from .chunking import ChunkValidation, MedicalParentChildChunker, approximate_tokens, normalize_document_text
from .context import RAGContextProvider, create_rag_search_tool
from .decomposition import LLMQueryDecomposer, QueryDecomposer, QueryDecomposition
from .ingestion import DocumentIndexer, RAGIngestionService
from .models import (
    Chunk, Citation, Document, DocumentStatus, Evidence, EvidenceBundle, EvidenceConflict,
    EvidenceStatus, IngestionResult, MedicalQuery, RetrievalFilters, RetrievalHit,
)
from .pipeline import RAGConfig, RAGPipeline, format_evidence_context
from .query import MedicalQueryPlanner
from .repository import InMemoryRAGRepository, RAGRepository, SQLiteRAGRepository, content_checksum
from .rerank import CallableReranker, HeuristicReranker, Reranker
from .retrieval import BM25Retriever, DenseRetriever, Retriever, reciprocal_rank_fusion, tokenize

__all__ = [
    "BM25Retriever", "CallableReranker", "Chunk", "ChunkValidation", "Citation",
    "DenseRetriever", "Document", "DocumentIndexer", "DocumentStatus", "Evidence",
    "EvidenceBundle", "EvidenceConflict", "EvidenceStatus", "HeuristicReranker",
    "InMemoryRAGRepository", "IngestionResult", "LLMQueryDecomposer",
    "MedicalParentChildChunker", "MedicalQuery", "MedicalQueryPlanner", "QueryDecomposer",
    "QueryDecomposition", "RAGConfig", "RAGContextProvider",
    "RAGIngestionService", "RAGPipeline", "RAGRepository", "Reranker", "Retriever",
    "RetrievalFilters", "RetrievalHit", "SQLiteRAGRepository", "approximate_tokens",
    "content_checksum", "create_rag_search_tool", "format_evidence_context",
    "normalize_document_text", "reciprocal_rank_fusion", "tokenize",
]
