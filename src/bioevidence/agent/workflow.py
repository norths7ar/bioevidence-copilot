from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bioevidence.config import Settings, load_settings
from bioevidence.extraction.evidence_extractor import extract_evidence
from bioevidence.generation.answerer import generate_answer
from bioevidence.ingestion.pubmed_client import search_pubmed
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.retrieval.hybrid import hybrid_retrieve
from bioevidence.retrieval.rerank import rerank_candidates
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    query: Query
    documents: tuple[Document, ...]
    retrieved_candidates: tuple[RetrievedCandidate, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    answer: AnswerBundle
    source: str


def run_rag_pipeline(
    query: Query,
    *,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> WorkflowResult:
    settings = settings or load_settings()
    documents = load_local_documents(data_dir or settings.data_dir, settings=settings)
    source = "local_corpus"
    if not documents:
        documents = search_pubmed(query, settings=settings)
        source = "pubmed_fallback"

    candidates = hybrid_retrieve(query, documents=documents, data_dir=data_dir, settings=settings)
    ranked_candidates = rerank_candidates(candidates)
    evidence_records = extract_evidence(query, [candidate.document for candidate in ranked_candidates[: query.top_k]])
    answer = generate_answer(query, evidence_records)
    return WorkflowResult(
        query=query,
        documents=tuple(documents),
        retrieved_candidates=tuple(ranked_candidates),
        evidence_records=tuple(evidence_records),
        answer=answer,
        source=source,
    )


def run_workflow(
    query: Query,
    *,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> AnswerBundle:
    return run_rag_pipeline(query, data_dir=data_dir, settings=settings).answer
