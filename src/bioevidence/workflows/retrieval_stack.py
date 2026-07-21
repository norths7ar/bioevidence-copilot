from __future__ import annotations

from pathlib import Path

from bioevidence.config import Settings, load_settings
from bioevidence.extraction.evidence_extractor import extract_evidence
from bioevidence.extraction.model_backend import ExtractionBackend
from bioevidence.ingestion.pubmed_client import search_pubmed
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.retrieval.hybrid import hybrid_retrieve
from bioevidence.retrieval.ranking import finalize_ranking
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def run_retrieval_stack(
    query: Query,
    *,
    data_dir: Path | None = None,
    documents: tuple[Document, ...] | list[Document] | None = None,
    settings: Settings | None = None,
    extraction_backend: ExtractionBackend | None = None,
) -> tuple[list[Document], list[RetrievedCandidate], list[EvidenceRecord], str]:
    settings = settings or load_settings()
    documents = list(documents) if documents is not None else load_local_documents(data_dir or settings.data_dir, settings=settings)
    source = "local_corpus"
    if not documents:
        documents = search_pubmed(query, settings=settings)
        source = "pubmed_fallback"

    candidates = hybrid_retrieve(query, documents=documents, data_dir=data_dir, settings=settings)
    ranked_candidates = finalize_ranking(candidates)
    evidence_records = extract_evidence(query, ranked_candidates[: query.top_k], backend=extraction_backend)
    return list(documents), list(ranked_candidates), list(evidence_records), source
