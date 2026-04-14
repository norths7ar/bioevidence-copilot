from __future__ import annotations

from dataclasses import dataclass, field

from bioevidence.schemas.evidence import EvidenceRecord


@dataclass(frozen=True, slots=True)
class AnswerBundle:
    answer_text: str
    citations: tuple[str, ...] = field(default_factory=tuple)
    evidence_records: tuple[EvidenceRecord, ...] = field(default_factory=tuple)
    rewritten_query: str | None = None
