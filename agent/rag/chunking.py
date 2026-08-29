"""Chinese-aware structure-first parent/child chunking."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Sequence

from .models import Chunk, Document


_HEADING_PATTERNS = (
    re.compile(r"^(#{1,6})\s+(.+)$"),
    re.compile(r"^(第[一二三四五六七八九十百零〇0-9]+[章节篇])\s*(.+)$"),
    re.compile(r"^([一二三四五六七八九十]+、)\s*(.+)$"),
    re.compile(r"^(（[一二三四五六七八九十0-9]+）)\s*(.+)$"),
    re.compile(r"^(\d+(?:\.\d+){0,3})[、.\s]+(.+)$"),
)
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|(?<=[.])\s+(?=[A-Z0-9])")
_PROTECTED_MARKERS = (
    "推荐意见", "推荐强度", "证据等级", "适应证", "禁忌", "用法用量",
    "剂量", "特殊人群", "警告", "注意事项", "不良反应",
)


def normalize_document_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def approximate_tokens(text: str) -> int:
    chinese = len(re.findall(r"[\u3400-\u9fff]", text))
    non_chinese = re.sub(r"[\u3400-\u9fff]", "", text)
    return max(1, chinese + max(0, len(non_chinese) // 4)) if text else 0


@dataclass
class Section:
    path: List[str]
    text: str
    start: int
    end: int
    units: List[str] = field(default_factory=list)


@dataclass
class ChunkValidation:
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class MedicalParentChildChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 250,
        min_tokens: int = 80,
        max_tokens: int = 400,
        version: str = "medical-parent-child-v1",
    ) -> None:
        if not 0 < min_tokens <= target_tokens <= max_tokens:
            raise ValueError("Require 0 < min_tokens <= target_tokens <= max_tokens.")
        self.target_tokens = target_tokens
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.version = version

    def chunk(self, document: Document) -> List[Chunk]:
        sections = self._sections(document.normalized_content, document.title)
        chunks: List[Chunk] = []
        sequence = 0
        for section in sections:
            parent_id = self._id(document.id, "parent", sequence, section.text)
            parent = Chunk(
                id=parent_id,
                document_id=document.id,
                text=section.text,
                contextual_text=self._context(document, section.path, section.text),
                section_path=section.path,
                sequence=sequence,
                chunk_type="parent",
                char_start=section.start,
                char_end=section.end,
                token_count=approximate_tokens(section.text),
                content_hash=hashlib.sha256(section.text.encode("utf-8")).hexdigest(),
                metadata={"chunker_version": self.version},
            )
            chunks.append(parent)
            sequence += 1
            for child_text in self._children(section):
                child_start = document.normalized_content.find(child_text, section.start, section.end)
                child_start = section.start if child_start < 0 else child_start
                chunk = Chunk(
                    id=self._id(document.id, "child", sequence, child_text),
                    document_id=document.id,
                    parent_chunk_id=parent_id,
                    text=child_text,
                    contextual_text=self._context(document, section.path, child_text),
                    section_path=section.path,
                    sequence=sequence,
                    chunk_type="child",
                    char_start=child_start,
                    char_end=child_start + len(child_text),
                    token_count=approximate_tokens(child_text),
                    evidence_grade=self._evidence_grade(child_text),
                    population=self._populations(child_text),
                    content_hash=hashlib.sha256(child_text.encode("utf-8")).hexdigest(),
                    metadata={"chunker_version": self.version},
                )
                chunks.append(chunk)
                sequence += 1
        return chunks

    def validate(self, document: Document, chunks: Sequence[Chunk]) -> ChunkValidation:
        errors: List[str] = []
        warnings: List[str] = []
        ids = {chunk.id for chunk in chunks}
        children = [chunk for chunk in chunks if chunk.chunk_type == "child"]
        if not children:
            errors.append("Document produced no child chunks.")
        for chunk in chunks:
            if not chunk.text.strip():
                errors.append(f"Chunk {chunk.id} is empty.")
            if chunk.document_id != document.id:
                errors.append(f"Chunk {chunk.id} has the wrong document id.")
            if chunk.chunk_type == "child" and chunk.parent_chunk_id not in ids:
                errors.append(f"Chunk {chunk.id} has no valid parent.")
            if chunk.char_start < 0 or chunk.char_end > len(document.normalized_content):
                warnings.append(f"Chunk {chunk.id} has approximate source offsets.")
            if chunk.chunk_type == "child" and chunk.token_count > self.max_tokens * 2:
                warnings.append(f"Protected chunk {chunk.id} is unusually large.")
        return ChunkValidation(not errors, errors, warnings)

    def _sections(self, text: str, title: str) -> List[Section]:
        path = [title]
        current_lines: List[str] = []
        sections: List[Section] = []
        cursor = 0
        section_start = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            heading = self._heading(stripped)
            if heading:
                if current_lines:
                    body = "".join(current_lines).strip()
                    if body:
                        sections.append(Section(list(path), body, section_start, cursor))
                level, name = heading
                path = path[: max(1, level)] + [name]
                current_lines = []
                section_start = cursor + len(line)
            elif stripped:
                current_lines.append(line)
            cursor += len(line)
        body = "".join(current_lines).strip()
        if body:
            sections.append(Section(list(path), body, section_start, len(text)))
        return sections or [Section([title], text, 0, len(text))]

    def _children(self, section: Section) -> List[str]:
        units = self._units(section.text)
        groups: List[List[str]] = []
        current: List[str] = []
        for unit in units:
            protected = self._protected(unit)
            projected = approximate_tokens("\n".join(current + [unit]))
            if current and (protected or projected > self.max_tokens):
                groups.append(current)
                current = []
            current.append(unit)
            if protected or approximate_tokens("\n".join(current)) >= self.target_tokens:
                groups.append(current)
                current = []
        if current:
            if groups and approximate_tokens("\n".join(current)) < self.min_tokens:
                groups[-1].extend(current)
            else:
                groups.append(current)
        return ["\n".join(group).strip() for group in groups if any(item.strip() for item in group)]

    def _units(self, text: str) -> List[str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        units: List[str] = []
        table_header = ""
        for line in lines:
            if "|" in line or "\t" in line:
                if not table_header:
                    table_header = line
                    continue
                if re.fullmatch(r"[|:\-\s]+", line):
                    continue
                units.append(f"表头: {table_header}\n表行: {line}")
                continue
            table_header = ""
            parts = [part.strip() for part in _SENTENCE_SPLIT.split(line) if part.strip()]
            for part in parts or [line]:
                # Evidence grade/strength qualifies the recommendation directly
                # before it, so never separate that condition-conclusion pair.
                if units and re.match(r"^(?:证据等级|推荐强度)\s*[:：]", part):
                    units[-1] = f"{units[-1]}{part}"
                else:
                    units.append(part)
        return units or [text]

    @staticmethod
    def _heading(line: str):
        for index, pattern in enumerate(_HEADING_PATTERNS):
            match = pattern.match(line)
            if match:
                level = len(match.group(1)) if index == 0 else min(index + 1, 5)
                return level, match.group(2).strip()
        return None

    @staticmethod
    def _protected(text: str) -> bool:
        return any(marker in text for marker in _PROTECTED_MARKERS) or bool(
            re.search(r"\b\d+(?:\.\d+)?\s*(?:mg|g|μg|ug|ml|mL|mmol|IU|单位)\b", text, re.I)
        )

    @staticmethod
    def _context(document: Document, path: Sequence[str], text: str) -> str:
        return (
            f"文档: {document.title}\n发布方: {document.publisher}\n"
            f"版本: {document.version}\n章节: {' > '.join(path)}\n正文: {text}"
        )

    @staticmethod
    def _id(document_id: str, kind: str, sequence: int, text: str) -> str:
        value = f"{document_id}\0{kind}\0{sequence}\0{text}".encode("utf-8")
        return hashlib.sha256(value).hexdigest()[:16]

    @staticmethod
    def _evidence_grade(text: str):
        match = re.search(r"证据等级\s*[:：]?\s*([A-DⅠⅡⅢIV1-4]+)", text, re.I)
        return match.group(1) if match else None

    @staticmethod
    def _populations(text: str) -> List[str]:
        mappings = {"儿童": "pediatric", "老年": "elderly", "妊娠": "pregnancy", "哺乳": "lactation", "成人": "adult"}
        return [value for term, value in mappings.items() if term in text]
