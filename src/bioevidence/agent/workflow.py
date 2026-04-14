from __future__ import annotations

from bioevidence.extraction.evidence_extractor import extract_evidence
from bioevidence.generation.answerer import generate_answer
from bioevidence.ingestion.pubmed_client import search_pubmed
from bioevidence.retrieval.hybrid import hybrid_retrieve
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.query import Query


def run_workflow(query: Query) -> AnswerBundle:
    documents = search_pubmed(query)
    candidates = hybrid_retrieve(query)
    evidence_records = extract_evidence(query, [candidate.document for candidate in candidates])

    if not evidence_records and documents:
        evidence_records = extract_evidence(query, documents)

    return generate_answer(query, evidence_records)
