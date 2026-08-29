"""Dependency-free demonstration of the governed hybrid RAG pipeline."""

from agent import (
    BM25Retriever, DenseRetriever, InMemoryRAGRepository,
    LLMEmbeddingProvider, MedicalParentChildChunker, MockLLM,
    RAGIngestionService, RAGPipeline, format_evidence_context,
)

repository = InMemoryRAGRepository()
embeddings = LLMEmbeddingProvider(MockLLM(), model_id="demo-only:hash-v1")
bm25 = BM25Retriever(repository)
dense = DenseRetriever(repository, embeddings)
ingestion = RAGIngestionService(
    repository, MedicalParentChildChunker(), [bm25, dense]
)

ingestion.ingest_text(
    logical_id="demo-guideline",
    title="示例高血压指南",
    publisher="示例学会",
    document_type="guideline",
    jurisdiction="CN",
    content="""# 成人治疗
推荐意见：成人患者应结合风险分层制定治疗方案。证据等级：A。
# 特殊人群
妊娠患者存在特定用药禁忌，应由专业医生评估。""",
)

pipeline = RAGPipeline(repository, bm25, dense)
print(format_evidence_context(pipeline.retrieve("妊娠患者用药需要注意什么？")))
