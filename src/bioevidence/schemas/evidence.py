from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    pmid: str
    title: str
    year: int | None
    journal: str
    entities: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""
    relevance_score: float = 0.0
