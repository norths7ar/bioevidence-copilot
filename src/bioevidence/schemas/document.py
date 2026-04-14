from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    pmid: str
    title: str = ""
    abstract: str = ""
    journal: str = ""
    year: int | None = None
    authors: tuple[str, ...] = field(default_factory=tuple)
    source: str = "pubmed"


@dataclass(frozen=True, slots=True)
class RetrievedCandidate:
    document: Document
    score: float = 0.0
    rank: int = 0
