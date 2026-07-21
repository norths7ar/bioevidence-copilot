from __future__ import annotations

import logging
from pathlib import Path

from bioevidence.config import Settings, load_settings
from bioevidence.extraction.model_backend import ExtractionBackend, create_product_extraction_backend
from bioevidence.generation.answerer import generate_answer
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.workflows.models import WorkflowResult
from bioevidence.workflows.retrieval_stack import run_retrieval_stack


LOGGER = logging.getLogger(__name__)


def run_rag_pipeline(
    query: Query,
    *,
    data_dir: Path | None = None,
    documents: tuple[Document, ...] | list[Document] | None = None,
    settings: Settings | None = None,
    extraction_backend: ExtractionBackend | None = None,
) -> WorkflowResult:
    settings = settings or load_settings()
    extraction_backend = extraction_backend or create_product_extraction_backend(settings)
    LOGGER.info("baseline_started top_k=%d", query.top_k)
    documents, ranked_candidates, evidence_records, source = run_retrieval_stack(
        query,
        data_dir=data_dir,
        documents=documents,
        settings=settings,
        extraction_backend=extraction_backend,
    )
    answer = generate_answer(query, evidence_records)
    LOGGER.info(
        "baseline_completed source=%s documents=%d candidates=%d evidence=%d citations=%d",
        source,
        len(documents),
        len(ranked_candidates),
        len(evidence_records),
        len(answer.citations),
    )
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
    documents: tuple[Document, ...] | list[Document] | None = None,
    settings: Settings | None = None,
) -> AnswerBundle:
    return run_rag_pipeline(query, data_dir=data_dir, documents=documents, settings=settings).answer
