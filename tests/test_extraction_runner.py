from __future__ import annotations

from bioevidence.evaluation.extraction_dataset import AnnotationStatus, ExtractionAnnotation
from bioevidence.evaluation.extraction_runner import run_extraction_evaluation
from bioevidence.extraction.model_backend import RuleBasedExtractionBackend
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction


def test_extraction_runner_summarizes_backend_attempts() -> None:
    document = Document(pmid="1", title="Asthma trial", abstract="Treatment reduced asthma symptoms.")
    expected = ModelEvidenceExtraction.model_validate(
        {
            "evidence_status": "direct",
            "study_design": "not_reported",
            "population_or_system": None,
            "intervention_or_exposure": None,
            "comparator": None,
            "outcomes": [
                {
                    "name": "reported query outcome",
                    "direction": "decreased",
                    "result_text": "Treatment reduced asthma symptoms.",
                    "evidence_span": "Treatment reduced asthma symptoms.",
                }
            ],
            "evidence_summary": "Treatment reduced asthma symptoms.",
        }
    )
    annotation = ExtractionAnnotation(
        id="item-1",
        query="Does treatment reduce asthma symptoms?",
        document=document,
        extraction=expected,
        annotation_status=AnnotationStatus.DRAFT,
    )

    report = run_extraction_evaluation([annotation], RuleBasedExtractionBackend())

    assert report.backend == "rules"
    assert report.summary["items"] == 1
    assert report.summary["json_parse_rate"] == 1.0
    assert report.summary["schema_validity_rate"] == 1.0
    assert report.summary["mean_evidence_status_accuracy"] == 1.0
