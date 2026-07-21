from __future__ import annotations

from bioevidence.extraction.table import evidence_table_rows, render_evidence_table
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.model_evidence import EvidenceStatus, ModelEvidenceExtraction, StudyDesign


def test_evidence_table_rows_preserves_expected_shape():
    records = [
        EvidenceRecord(
            pmid="1",
            title="Corticosteroids for asthma control",
            year=2024,
            journal="Journal A",
            entities=("asthma", "corticosteroids"),
            summary="Corticosteroids reduce asthma exacerbations and improve control",
            relevance_score=0.91,
        )
    ]

    rows = evidence_table_rows(records)

    assert rows == [
        {
            "pmid": "1",
            "title": "Corticosteroids for asthma control",
            "year": 2024,
            "journal": "Journal A",
            "entities": ["asthma", "corticosteroids"],
            "summary": "Corticosteroids reduce asthma exacerbations and improve control",
            "relevance_score": 0.91,
        }
    ]


def test_render_evidence_table_produces_human_readable_ascii():
    records = [
        EvidenceRecord(
            pmid="1",
            title="Corticosteroids for asthma control",
            year=2024,
            journal="Journal A",
            entities=("asthma", "corticosteroids"),
            summary="Corticosteroids reduce asthma exacerbations and improve control",
            relevance_score=0.91,
        )
    ]

    rendered = render_evidence_table(records)

    assert "PMID" in rendered
    assert "Corticosteroids for asthma control" in rendered
    assert "asthma, corticosteroids" in rendered
    assert "0.9100" in rendered


def test_evidence_table_rows_surface_optional_model_extraction():
    record = EvidenceRecord(
        pmid="1",
        title="Protocol",
        year=2024,
        journal="Journal",
        model_extraction=ModelEvidenceExtraction(
            evidence_status=EvidenceStatus.NONE,
            study_design=StudyDesign.STUDY_PROTOCOL,
            population_or_system=None,
            intervention_or_exposure=None,
            comparator=None,
            outcomes=(),
            evidence_summary=None,
        ),
    )

    row = evidence_table_rows([record])[0]

    assert row["evidence_status"] == "none"
    assert row["study_design"] == "study_protocol"
    assert row["outcomes"] == []
