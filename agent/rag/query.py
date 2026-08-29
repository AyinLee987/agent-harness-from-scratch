"""Deterministic medical query normalization and lightweight entity extraction."""

from __future__ import annotations

import re
import unicodedata

from .models import MedicalQuery, RetrievalFilters


class MedicalQueryPlanner:
    def plan(self, text: str, filters: RetrievalFilters | None = None) -> MedicalQuery:
        normalized = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()
        if not normalized:
            raise ValueError("Query cannot be empty.")
        entities = list(dict.fromkeys(re.findall(
            r"[A-Za-z][A-Za-z0-9+.-]{2,}|\d+(?:\.\d+)?\s*(?:mg|g|μg|ml|mmol|IU)|"
            r"(?:儿童|老年人?|孕妇|妊娠|哺乳期|成人|肝功能不全|肾功能不全)",
            normalized, re.I,
        )))
        subquestions = [part.strip() for part in re.split(r"[？?；;]", normalized) if part.strip()]
        return MedicalQuery(
            original=text,
            normalized=normalized,
            lexical_queries=[normalized] + entities,
            semantic_queries=[normalized],
            entities=entities,
            filters=filters or RetrievalFilters(),
            subquestions=subquestions if len(subquestions) > 1 else [],
        )
