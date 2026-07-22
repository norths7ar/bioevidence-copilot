from __future__ import annotations

import json
from pathlib import Path

import scripts.diagnose_extraction as diagnose_script
from bioevidence.extraction.model_backend import ExtractionBackendError, FallbackExtractionBackend
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import EvidenceStatus, ModelEvidenceExtraction, StudyDesign


class _InvalidBackend:
    name = "local_adapter"

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction:
        raise ExtractionBackendError(
            "invalid schema",
            kind="schema",
            raw_output='{"evidence_status": "invalid"}',
            details='[{"loc": ["evidence_status"]}]',
        )


class _FallbackBackend:
    name = "rules"

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction:
        return ModelEvidenceExtraction(
            evidence_status=EvidenceStatus.NONE,
            study_design=StudyDesign.NOT_REPORTED,
            population_or_system=None,
            intervention_or_exposure=None,
            comparator=None,
            outcomes=(),
            evidence_summary=None,
        )


def test_diagnostic_cli_writes_failure_provenance(tmp_path: Path, monkeypatch, capsys) -> None:
    output = tmp_path / "diagnostic.json"
    document = Document(pmid="123", title="Trial", abstract="Abstract")
    backend = FallbackExtractionBackend(_InvalidBackend(), _FallbackBackend())
    monkeypatch.setattr(diagnose_script, "create_product_extraction_backend", lambda settings: backend)
    monkeypatch.setattr(diagnose_script, "load_local_documents", lambda data_dir, settings: [document])

    exit_code = diagnose_script.main(
        ["--query", "asthma trial", "--pmid", "123", "--output", str(output)]
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["attempted_backend"] == "local_adapter"
    assert report["used_backend"] == "rules"
    assert report["fallback_reason"] == "schema"
    assert report["failure_details"] == '[{"loc": ["evidence_status"]}]'
    assert report["failed_raw_output"] == '{"evidence_status": "invalid"}'
    assert "Report:" in capsys.readouterr().out
