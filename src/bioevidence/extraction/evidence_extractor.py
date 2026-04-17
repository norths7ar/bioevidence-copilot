from __future__ import annotations

from bioevidence.schemas.document import Document
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def extract_evidence(query: Query, documents: list[Document]) -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for index, document in enumerate(documents):
        summary = document.abstract[:200].strip()
        if not summary:
            summary = document.title[:200].strip()
        records.append(
            EvidenceRecord(
                pmid=document.pmid,
                title=document.title,
                year=document.year,
                journal=document.journal,
                entities=tuple(token for token in query.text.split() if token),
                summary=summary,
                relevance_score=max(0.0, 1.0 - (index * 0.1)),
            )
        )
    return records
