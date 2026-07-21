from __future__ import annotations

from bioevidence.evaluation.extraction_metrics import compute_extraction_metrics
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction


def _extraction(*, direction: str = "decreased", span: str = "Treatment reduced symptoms.") -> ModelEvidenceExtraction:
    return ModelEvidenceExtraction.model_validate(
        {
            "evidence_status": "direct",
            "study_design": "randomized_controlled_trial",
            "population_or_system": "adults with asthma",
            "intervention_or_exposure": "treatment",
            "comparator": "placebo",
            "outcomes": [
                {
                    "name": "symptoms",
                    "direction": direction,
                    "result_text": span,
                    "evidence_span": span,
                }
            ],
            "evidence_summary": span,
        }
    )


def test_extraction_metrics_are_one_for_identical_predictions() -> None:
    expected = _extraction()

    metrics = compute_extraction_metrics(expected, expected, abstract="Treatment reduced symptoms.")

    assert all(value == 1.0 for value in metrics.values())


def test_extraction_metrics_penalize_direction_and_unsupported_span() -> None:
    expected = _extraction()
    predicted = _extraction(direction="increased", span="Unsupported sentence.")

    metrics = compute_extraction_metrics(predicted, expected, abstract="Treatment reduced symptoms.")

    assert metrics["outcome_direction_accuracy"] == 0.0
    assert metrics["evidence_span_support_rate"] == 0.0
