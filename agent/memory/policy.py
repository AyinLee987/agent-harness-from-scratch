"""Fail-closed memory extraction and persistence policies."""

from __future__ import annotations

import re
from typing import List, Protocol

from .models import (
    MemoryCandidate,
    MemoryDecision,
    MemoryKind,
    RetentionPolicy,
    RunCompletedEvent,
    Sensitivity,
)


class MemoryExtractor(Protocol):
    def extract(self, event: RunCompletedEvent) -> List[MemoryCandidate]:
        ...


class MemoryPolicy(Protocol):
    def decide(self, candidate: MemoryCandidate) -> MemoryDecision:
        ...


class NoopMemoryExtractor:
    """Safe default: no model output or conversation is persisted implicitly."""

    def extract(self, event: RunCompletedEvent) -> List[MemoryCandidate]:
        return []


class ExplicitRequestMemoryExtractor:
    """Opt-in extractor for direct “remember …” user requests.

    This intentionally does not infer preferences or medical facts.  It stores
    the user's own statement as a candidate and leaves approval to the policy.
    """

    _patterns = (
        re.compile(r"(?:please\s+)?remember(?:\s+that)?\s*[:：]?\s*(.+)", re.I | re.S),
        re.compile(r"(?:请)?记住\s*[:：]?\s*(.+)", re.S),
    )

    def extract(self, event: RunCompletedEvent) -> List[MemoryCandidate]:
        if not event.success:
            return []
        content = ""
        for pattern in self._patterns:
            match = pattern.search(event.task.strip())
            if match:
                content = match.group(1).strip()
                break
        if not content:
            return []
        sensitivity = _classify_sensitivity(content)
        return [
            MemoryCandidate(
                content=content,
                kind=MemoryKind.USER_FACT,
                namespace=event.namespace,
                subject_id=event.subject_id,
                source_type="user_message",
                source_ref=f"run:{event.run_id}",
                source_run_id=event.run_id,
                confidence=1.0,
                importance=0.7,
                sensitivity=sensitivity,
                verification_status="user_stated",
                retention_policy=RetentionPolicy.EXPLICIT_DELETE_ONLY,
                explicit_user_request=True,
            )
        ]


class DefaultMemoryPolicy:
    """Conservative policy that never trusts model-generated facts."""

    def decide(self, candidate: MemoryCandidate) -> MemoryDecision:
        if not candidate.content.strip():
            return MemoryDecision.SKIP
        if candidate.source_type in {"model_output", "model_inference"}:
            return MemoryDecision.SKIP
        if candidate.sensitivity in {Sensitivity.HEALTH, Sensitivity.SECRET}:
            return MemoryDecision.REQUIRE_CONFIRMATION
        if candidate.retention_policy in {
            RetentionPolicy.EPHEMERAL,
            RetentionPolicy.TTL,
        }:
            return MemoryDecision.EPHEMERAL
        if candidate.explicit_user_request or candidate.metadata.get("trusted_source"):
            return MemoryDecision.PERSIST
        return MemoryDecision.REQUIRE_CONFIRMATION


def _classify_sensitivity(content: str) -> Sensitivity:
    lowered = content.casefold()
    health_terms = (
        "allergy",
        "diagnosis",
        "disease",
        "medicine",
        "medical",
        "过敏",
        "疾病",
        "诊断",
        "病史",
        "用药",
        "药物",
    )
    secret_terms = ("password", "api key", "token", "密码", "密钥", "身份证")
    if any(term in lowered for term in secret_terms):
        return Sensitivity.SECRET
    if any(term in lowered for term in health_terms):
        return Sensitivity.HEALTH
    return Sensitivity.NORMAL
