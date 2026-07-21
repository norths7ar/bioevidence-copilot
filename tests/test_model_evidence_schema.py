import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from bioevidence.schemas.model_evidence import (
    ModelEvidenceExtraction,
    OutcomeEvidence,
    unsupported_evidence_spans,
)


def _direct_extraction() -> ModelEvidenceExtraction:
    return ModelEvidenceExtraction(
        evidence_status="direct",
        study_design="randomized_controlled_trial",
        population_or_system="adults with hypertension",
        intervention_or_exposure="salt substitute",
        comparator="regular salt",
        outcomes=(
            OutcomeEvidence(
                name="systolic blood pressure",
                direction="decreased",
                result_text="The salt substitute reduced systolic blood pressure.",
                evidence_span="The salt substitute reduced systolic blood pressure.",
            ),
        ),
        evidence_summary="A randomized trial found lower blood pressure with the salt substitute.",
    )


def test_model_evidence_schema_accepts_grounded_direct_evidence() -> None:
    extraction = _direct_extraction()

    assert extraction.evidence_status == "direct"
    assert unsupported_evidence_spans(
        extraction,
        "RESULTS: The salt substitute reduced systolic blood pressure.",
    ) == ()


def test_model_evidence_schema_rejects_direct_evidence_without_outcome() -> None:
    with pytest.raises(ValidationError, match="direct evidence requires at least one grounded outcome"):
        ModelEvidenceExtraction(
            evidence_status="direct",
            study_design="cohort",
            population_or_system="adults",
            intervention_or_exposure="metformin",
            comparator=None,
            outcomes=(),
            evidence_summary="Metformin was evaluated.",
        )


def test_model_evidence_schema_rejects_content_for_none_status() -> None:
    with pytest.raises(ValidationError, match="none evidence must not contain extracted evidence fields"):
        ModelEvidenceExtraction(
            evidence_status="none",
            study_design="cohort",
            population_or_system="adults",
            intervention_or_exposure=None,
            comparator=None,
            outcomes=(),
            evidence_summary=None,
        )


def test_unsupported_evidence_spans_reports_non_verbatim_text() -> None:
    extraction = _direct_extraction()

    assert unsupported_evidence_spans(extraction, "The abstract reports a different result.") == (
        "The salt substitute reduced systolic blood pressure.",
    )


def test_checked_in_json_schema_matches_runtime_contract() -> None:
    schema_path = Path("schemas/model_evidence_extraction.v1.schema.json")

    assert json.loads(schema_path.read_text(encoding="utf-8")) == ModelEvidenceExtraction.model_json_schema()
