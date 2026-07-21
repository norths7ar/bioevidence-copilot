from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import scripts.run_extraction_eval as run_script
from bioevidence.evaluation.extraction_runner import ExtractionEvaluationReport


def test_run_extraction_eval_cli_writes_report(tmp_path: Path, capsys, monkeypatch) -> None:
    output = tmp_path / "report.json"
    report = ExtractionEvaluationReport(
        backend="rules",
        generated_at=datetime.now(timezone.utc),
        summary={
            "items": 1,
            "json_parse_rate": 1.0,
            "schema_validity_rate": 1.0,
            "mean_latency_ms": 0.1,
            "mean_evidence_status_accuracy": 1.0,
            "mean_study_design_accuracy": 1.0,
            "mean_semantic_field_token_f1": 1.0,
            "mean_outcome_name_token_f1": 1.0,
            "mean_outcome_direction_accuracy": 1.0,
            "mean_evidence_span_token_f1": 1.0,
            "mean_evidence_span_support_rate": 1.0,
        },
    )
    monkeypatch.setattr(run_script, "load_dotenv", lambda: None)
    monkeypatch.setattr(run_script, "load_local_documents", lambda path: [])
    monkeypatch.setattr(run_script, "load_extraction_annotations", lambda path, documents: [])
    monkeypatch.setattr(run_script, "run_extraction_evaluation", lambda annotations, backend, limit: report)

    exit_code = run_script.main(["--backend", "rules", "--output", str(output)])

    assert exit_code == 0
    assert "Evidence extraction evaluation" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["backend"] == "rules"
