from __future__ import annotations

import re
from dataclasses import dataclass, field
from collections.abc import Iterable, Sequence

from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.evidence import EvidenceRecord


_PMID_CITATION_RE = re.compile(r"\[(\d{2,})\]")
_INSUFFICIENT_MARKERS = (
    "insufficient evidence",
    "no local evidence",
    "no evidence",
    "not enough evidence",
)


@dataclass(frozen=True, slots=True)
class EvidenceMetadata:
    pmid: str
    study_type: str = "unspecified"
    effect_direction: str = "unspecified"

    def to_dict(self) -> dict[str, str]:
        return {
            "pmid": self.pmid,
            "study_type": self.study_type,
            "effect_direction": self.effect_direction,
        }


@dataclass(frozen=True, slots=True)
class QualityCheckResult:
    has_evidence: bool
    is_faithful: bool
    insufficient_evidence: bool
    forced_conclusion_without_evidence: bool
    supported_citations: tuple[str, ...] = field(default_factory=tuple)
    unsupported_citations: tuple[str, ...] = field(default_factory=tuple)
    missing_citations: tuple[str, ...] = field(default_factory=tuple)
    evidence_pmids: tuple[str, ...] = field(default_factory=tuple)
    inline_citations: tuple[str, ...] = field(default_factory=tuple)
    evidence_metadata: tuple[EvidenceMetadata, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "has_evidence": self.has_evidence,
            "is_faithful": self.is_faithful,
            "insufficient_evidence": self.insufficient_evidence,
            "forced_conclusion_without_evidence": self.forced_conclusion_without_evidence,
            "supported_citations": list(self.supported_citations),
            "unsupported_citations": list(self.unsupported_citations),
            "missing_citations": list(self.missing_citations),
            "evidence_pmids": list(self.evidence_pmids),
            "inline_citations": list(self.inline_citations),
            "evidence_metadata": [item.to_dict() for item in self.evidence_metadata],
            "notes": list(self.notes),
        }


def check_answer_quality(
    answer: AnswerBundle,
    evidence_records: Sequence[EvidenceRecord],
) -> QualityCheckResult:
    evidence_pmids = _unique(record.pmid for record in evidence_records if record.pmid)
    evidence_set = set(evidence_pmids)
    answer_citations = _unique(answer.citations)
    inline_citations = _unique(_PMID_CITATION_RE.findall(answer.answer_text))

    supported_citations = tuple(citation for citation in answer_citations if citation in evidence_set)
    unsupported_citations = tuple(citation for citation in answer_citations if citation not in evidence_set)
    missing_citations = tuple(citation for citation in inline_citations if citation not in answer_citations)
    has_evidence = bool(evidence_pmids)
    insufficient_evidence = not has_evidence
    forced_conclusion = insufficient_evidence and not _states_insufficient_evidence(answer.answer_text)

    notes: list[str] = []
    if unsupported_citations:
        notes.append("answer cites PMIDs that are not present in the evidence table")
    if missing_citations:
        notes.append("answer text contains inline PMID citations missing from the citation list")
    if forced_conclusion:
        notes.append("answer gives a conclusion even though no evidence records were available")
    if has_evidence and not answer_citations:
        notes.append("evidence was available but the answer did not return citations")

    is_faithful = not unsupported_citations and not missing_citations and not forced_conclusion
    if has_evidence and not answer_citations:
        is_faithful = False

    return QualityCheckResult(
        has_evidence=has_evidence,
        is_faithful=is_faithful,
        insufficient_evidence=insufficient_evidence,
        forced_conclusion_without_evidence=forced_conclusion,
        supported_citations=supported_citations,
        unsupported_citations=unsupported_citations,
        missing_citations=missing_citations,
        evidence_pmids=evidence_pmids,
        inline_citations=inline_citations,
        evidence_metadata=tuple(_derive_evidence_metadata(record) for record in evidence_records),
        notes=tuple(notes),
    )


def _derive_evidence_metadata(record: EvidenceRecord) -> EvidenceMetadata:
    text = f"{record.title} {record.summary}".lower()
    study_type = "unspecified"
    if "meta-analysis" in text or "systematic review" in text:
        study_type = "review"
    elif "randomized" in text or "randomised" in text or "trial" in text:
        study_type = "trial"
    elif "cohort" in text or "case-control" in text or "observational" in text:
        study_type = "observational"

    effect_direction = "unspecified"
    if any(marker in text for marker in ("reduce", "reduced", "improve", "improved", "benefit")):
        effect_direction = "benefit"
    elif any(marker in text for marker in ("increase risk", "increased risk", "harm", "adverse")):
        effect_direction = "harm"
    elif any(marker in text for marker in ("mixed", "conflicting", "inconsistent")):
        effect_direction = "mixed"
    elif any(marker in text for marker in ("no significant", "not associated", "neutral")):
        effect_direction = "neutral"

    return EvidenceMetadata(
        pmid=record.pmid,
        study_type=study_type,
        effect_direction=effect_direction,
    )


def _states_insufficient_evidence(text: str) -> bool:
    normalized = text.lower()
    return any(marker in normalized for marker in _INSUFFICIENT_MARKERS)


def _unique(values: Iterable[str] | object) -> tuple[str, ...]:
    seen: set[str] = set()
    unique_values: list[str] = []
    iterable = () if isinstance(values, str) else values
    for value in iterable:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_values.append(normalized)
    return tuple(unique_values)
