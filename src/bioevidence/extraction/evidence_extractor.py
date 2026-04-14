from __future__ import annotations

from bioevidence.schemas.document import Document
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def extract_evidence(query: Query, documents: list[Document]) -> list[EvidenceRecord]:
    _ = query
    records: list[EvidenceRecord] = []
    for document in documents:
        records.append(
            EvidenceRecord(
                pmid=document.pmid,
                title=document.title,
                year=document.year,
                journal=document.journal,
                summary=document.abstract[:200],
                relevance_score=0.0,
            )
        )
    return records
