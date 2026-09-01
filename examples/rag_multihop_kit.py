"""A tiny synthetic medical corpus for measuring whether query decomposition
(agent/rag/decomposition.py) actually helps parallel-composite and
sequential-dependent ("multi-hop") questions, against a real LLM.

Design: five fact-pairs, each split across two short documents on purpose,
in one of two shapes:

    parallel   -- both documents are directly about the same drug; the
                  question just asks two independent things about it
                  ("is it safe in pregnancy, and what's the adult dose").
    sequential -- the second document's fact only becomes findable once you
                  know *which* drug the first document names (e.g. "the
                  first-line drug for condition X" -> only then can you look
                  up "condition X's drug's dose in the elderly"). The
                  question never names the drug -- that's the whole point.

See rag_multihop_tasks.json for the actual questions and
rag_multihop_eval.py for the baseline-vs-decomposer comparison.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # repo root -> `import agent`

from agent import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    InMemoryRAGRepository,
    MedicalParentChildChunker,
    RAGConfig,
    RAGContextProvider,
    RAGIngestionService,
    RAGPipeline,
    ReActAgent,
    ToolRegistry,
    create_rag_search_tool,
)
from agent.llm import BaseLLM
from agent.memory.embeddings import EmbeddingProvider
from agent.rag.decomposition import QueryDecomposer

DOCS = [
    # parallel_analgesic
    dict(logical_id="analgesic-pregnancy", title="妊娠禁忌",
         content="# 妊娠禁忌\n妊娠患者禁忌使用镇痛药X。"),
    dict(logical_id="analgesic-dose", title="成人剂量",
         content="# 成人剂量\n镇痛药X成人推荐剂量为每次400mg，每日不超过3次。"),
    # parallel_antipyretic
    dict(logical_id="antipyretic-neonate", title="儿童禁忌",
         content="# 儿童禁忌\n退烧药Z禁止用于新生儿。"),
    dict(logical_id="antipyretic-dose", title="退烧药用法",
         content="# 退烧药用法\n退烧药Z成人推荐剂量为每次500mg，间隔不少于4小时。"),
    # parallel_bisphosphonate
    dict(logical_id="bisphosphonate-renal", title="骨质疏松禁忌",
         content="# 骨质疏松禁忌\n肾功能不全患者禁忌使用双膦酸盐类药物M。"),
    dict(logical_id="bisphosphonate-dose", title="双膦酸盐用法",
         content="# 双膦酸盐用法\n双膦酸盐类药物M成人推荐剂量为每周一次70mg，晨起空腹服用。"),
    # sequential_htn_elderly -- question never says "降压药Y"
    dict(logical_id="htn-firstline", title="一线降压药",
         content="# 一线降压药\n成人高血压一线推荐用药为降压药Y。证据等级：A。"),
    dict(logical_id="htn-elderly-dose", title="老年剂量",
         content="# 老年剂量\n降压药Y在老年人群体的推荐剂量为每日2.5mg，从低剂量起始。"),
    # sequential_pneumonia_pregnancy -- question never says "抗生素Q"
    dict(logical_id="pneumonia-firstline", title="感染一线用药",
         content="# 感染一线用药\n成人社区获得性肺炎一线推荐使用抗生素Q。证据等级：A。"),
    dict(logical_id="antibiotic-pregnancy-grade", title="抗生素妊娠分级",
         content="# 抗生素妊娠分级\n抗生素Q妊娠期分级为C级，需权衡利弊使用。"),
]

# Pure distractors -- unrelated drugs/conditions sharing little vocabulary
# with the fact-pairs above. Without these, a 10-document corpus is small
# enough that RAGConfig.evidence_limit lets almost everything through
# regardless of query, which makes "does decomposition help retrieval
# precision" untestable (verified: with no distractors, a plain single-shot
# query already pulls back nearly the whole corpus). These exist purely to
# give retrieval something real to filter out.
_DISTRACTOR_TOPICS = [
    ("甲状腺功能减退治疗", "成人甲状腺功能减退首选左甲状腺素钠替代治疗，需定期监测促甲状腺激素水平。"),
    ("糖尿病用药调整", "二甲双胍为成人2型糖尿病一线用药，肾功能不全患者需调整剂量或停用。"),
    ("哮喘急性发作处理", "哮喘急性发作首选吸入短效β2受体激动剂，症状缓解后评估是否需要升级治疗。"),
    ("抗凝药物监测", "使用华法林抗凝治疗期间需定期监测国际标准化比值，维持在目标范围内。"),
    ("癫痫用药选择", "成人局灶性癫痫一线用药包括卡马西平和左乙拉西坦，需根据合并用药调整。"),
    ("痛风急性期治疗", "痛风急性发作期首选非甾体抗炎药或秋水仙碱，肾功能不全者需谨慎选择剂量。"),
    ("抑郁症药物治疗", "成人中重度抑郁症一线推荐选择性5-羟色胺再摄取抑制剂，起效通常需要2至4周。"),
    ("骨关节炎疼痛管理", "骨关节炎轻中度疼痛可局部使用非甾体抗炎药凝胶，全身用药需评估心血管风险。"),
    ("偏头痛预防用药", "偏头痛发作频繁患者可考虑普萘洛尔或托吡酯作为预防性用药，需评估禁忌症。"),
    ("慢性阻塞性肺疾病管理", "慢性阻塞性肺疾病稳定期首选长效支气管扩张剂，急性加重期需评估是否使用抗生素。"),
    ("骨折术后康复", "骨折内固定术后早期康复应循序渐进，避免过早负重导致内固定失效。"),
    ("急性胃肠炎补液", "急性胃肠炎伴脱水首选口服补液盐，重度脱水或呕吐严重者需静脉补液。"),
    # These two specifically compete on "老年"+"剂量" lexical overlap with
    # htn-elderly-dose (verified: without them, a bare single-shot query
    # for "一线用药在老年人的推荐剂量" finds htn-elderly-dose by keyword
    # overlap alone, without ever needing to know the drug is 降压药Y --
    # not a real test of the sequential-dependency problem this task is
    # supposed to probe).
    ("他汀类药物老年人剂量", "老年人使用他汀类药物降脂治疗需从低剂量起始，定期监测肝功能。"),
    ("抗血小板药物老年人剂量", "老年患者使用抗血小板药物阿司匹林需评估出血风险，必要时减量。"),
]
DOCS.extend(
    dict(logical_id=f"distractor-{index}", title=title, content=f"# {title}\n{content}")
    for index, (title, content) in enumerate(_DISTRACTOR_TOPICS)
)


def build_pipeline(
    embeddings: EmbeddingProvider, *, decomposer: Optional[QueryDecomposer] = None,
) -> RAGPipeline:
    """A fresh in-memory corpus with all DOCS ingested, ready to query."""
    repository = InMemoryRAGRepository()
    bm25 = BM25Retriever(repository)
    dense = DenseRetriever(repository, embeddings)
    ingestion = RAGIngestionService(
        repository,
        MedicalParentChildChunker(target_tokens=60, min_tokens=15, max_tokens=100),
        [bm25, dense],
    )
    for doc in DOCS:
        ingestion.ingest_text(**doc)
    # A small evidence_limit relative to the corpus (22 documents once
    # distractors are included) is what makes retrieval actually selective
    # -- a generous limit against a small corpus lets nearly everything
    # through regardless of query, which defeats the point of this eval.
    config = RAGConfig(minimum_evidence=1, evidence_limit=4)
    return RAGPipeline(repository, bm25, dense, config=config, decomposer=decomposer)


def build_agent(llm_factory: Callable[[], BaseLLM], pipeline: RAGPipeline, *, max_steps: int = 8) -> ReActAgent:
    """A Leader-shaped agent: mandatory RAG injection + the follow-up
    medical_evidence_search tool, same wiring app/server.py uses."""
    tools = ToolRegistry([create_rag_search_tool(pipeline)])
    return ReActAgent(
        llm=llm_factory(),
        tools=tools,
        context_providers=[RAGContextProvider(pipeline)],
        max_steps=max_steps,
    )
