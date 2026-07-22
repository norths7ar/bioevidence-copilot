from __future__ import annotations

from dataclasses import dataclass, field

from bioevidence.schemas.model_evidence import ModelEvidenceExtraction


@dataclass(frozen=True, slots=True)
class ExtractionProvenance:
    attempted_backend: str
    used_backend: str
    fallback_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    pmid: str
    title: str
    year: int | None
    journal: str
    entities: tuple[str, ...] = field(default_factory=tuple)
    summary: str = ""
    relevance_score: float = 0.0
    model_extraction: ModelEvidenceExtraction | None = None
    extraction_provenance: ExtractionProvenance | None = None
