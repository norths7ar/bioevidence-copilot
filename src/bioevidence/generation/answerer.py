from __future__ import annotations

from bioevidence.generation.citation_formatter import format_citations
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def generate_answer(query: Query, evidence_records: list[EvidenceRecord]) -> AnswerBundle:
    citations = tuple(record.pmid for record in evidence_records)
    citation_text = format_citations(citations)
    answer_text = (
        "Scaffold only: no real synthesis has been implemented yet."
        if not evidence_records
        else f"Scaffold answer for '{query.text}'. Citations: {citation_text}"
    )
    return AnswerBundle(
        answer_text=answer_text,
        citations=citations,
        evidence_records=tuple(evidence_records),
        rewritten_query=query.rewritten_text,
    )
