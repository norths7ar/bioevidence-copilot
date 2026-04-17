from __future__ import annotations

from bioevidence.generation.citation_formatter import format_citations
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def generate_answer(query: Query, evidence_records: list[EvidenceRecord]) -> AnswerBundle:
    citations = tuple(record.pmid for record in evidence_records)
    citation_text = format_citations(citations)
    if not evidence_records:
        answer_text = f"No local evidence was retrieved for '{query.text}'."
    else:
        lead_records = evidence_records[:3]
        lead_sentences = []
        for record in lead_records:
            year_text = str(record.year) if record.year is not None else "n.d."
            summary = record.summary.strip().rstrip(".")
            citation = f"[{record.pmid}]"
            lead_sentences.append(f"{record.title} ({year_text}) suggests {summary} {citation}".strip())
        answer_text = f"Top retrieved evidence for '{query.text}': " + " ".join(lead_sentences)
        if citation_text:
            answer_text = f"{answer_text} Citations: {citation_text}"
    return AnswerBundle(
        answer_text=answer_text,
        citations=citations,
        evidence_records=tuple(evidence_records),
        rewritten_query=query.rewritten_text or query.text,
    )
