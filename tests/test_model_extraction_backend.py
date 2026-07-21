from __future__ import annotations

import json

from bioevidence.extraction.model_backend import (
    PromptedExtractionBackend,
    RuleBasedExtractionBackend,
    build_extraction_messages,
    run_extraction_attempt,
)
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import EvidenceStatus, StudyDesign


def test_prompted_backend_validates_grounded_json() -> None:
    document = Document(pmid="1", title="Trial", abstract="Treatment reduced symptom scores.")
    payload = {
        "evidence_status": "direct",
        "study_design": "randomized_controlled_trial",
        "population_or_system": "patients",
        "intervention_or_exposure": "treatment",
        "comparator": None,
        "outcomes": [
            {
                "name": "symptom scores",
                "direction": "decreased",
                "result_text": "Treatment reduced symptom scores.",
                "evidence_span": "Treatment reduced symptom scores.",
            }
        ],
        "evidence_summary": "Treatment reduced symptom scores.",
    }
    backend = PromptedExtractionBackend(
        api_key="",
        base_url="",
        model="test-model",
        completion=lambda messages: f"```json\n{json.dumps(payload)}\n```",
    )

    extraction = backend.extract("Does treatment reduce symptoms?", document)

    assert extraction.evidence_status is EvidenceStatus.DIRECT
    assert extraction.outcomes[0].evidence_span in document.abstract


def test_prompted_backend_reports_json_failure() -> None:
    backend = PromptedExtractionBackend(
        api_key="",
        base_url="",
        model="test-model",
        completion=lambda messages: "not json",
    )

    attempt = run_extraction_attempt(backend, "query", Document(pmid="1", abstract="Abstract"))

    assert attempt.error_kind == "json"
    assert attempt.json_parsed is False
    assert attempt.schema_valid is False


def test_rule_backend_returns_schema_valid_protocol_none() -> None:
    backend = RuleBasedExtractionBackend()
    document = Document(
        pmid="1",
        title="A randomized trial protocol for asthma",
        abstract="This study protocol describes an asthma intervention.",
    )

    extraction = backend.extract("Does the intervention improve asthma?", document)

    assert extraction.evidence_status is EvidenceStatus.NONE
    assert extraction.study_design is StudyDesign.STUDY_PROTOCOL


def test_prompt_contains_schema_and_source_boundaries() -> None:
    messages = build_extraction_messages("query", Document(pmid="1", title="title", abstract="abstract"))

    assert len(messages) == 2
    assert "evidence_status" in messages[1]["content"]
    assert "QUERY:\nquery" in messages[1]["content"]
    assert "ABSTRACT:\nabstract" in messages[1]["content"]
