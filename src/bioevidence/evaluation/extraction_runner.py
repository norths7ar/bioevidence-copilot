from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bioevidence.evaluation.extraction_dataset import ExtractionAnnotation
from bioevidence.evaluation.extraction_metrics import compute_extraction_metrics, mean_metrics
from bioevidence.extraction.model_backend import ExtractionBackend, ExtractionAttempt, run_extraction_attempt


@dataclass(frozen=True, slots=True)
class ExtractionEvaluationItem:
    annotation_id: str
    pmid: str
    attempt: ExtractionAttempt
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "annotation_id": self.annotation_id,
            "pmid": self.pmid,
            "prediction": self.attempt.extraction.model_dump(mode="json") if self.attempt.extraction else None,
            "latency_ms": round(self.attempt.latency_ms, 3),
            "json_parsed": self.attempt.json_parsed,
            "schema_valid": self.attempt.schema_valid,
            "error_kind": self.attempt.error_kind,
            "error_message": self.attempt.error_message,
            "raw_output": self.attempt.raw_output,
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class ExtractionEvaluationReport:
    backend: str
    generated_at: datetime
    summary: dict[str, float | int]
    items: tuple[ExtractionEvaluationItem, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
        }


def run_extraction_evaluation(
    annotations: list[ExtractionAnnotation],
    backend: ExtractionBackend,
    *,
    limit: int | None = None,
) -> ExtractionEvaluationReport:
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be a positive integer")
        annotations = annotations[:limit]

    results: list[ExtractionEvaluationItem] = []
    for annotation in annotations:
        attempt = run_extraction_attempt(backend, annotation.query, annotation.document)
        metrics = compute_extraction_metrics(
            attempt.extraction,
            annotation.extraction,
            abstract=annotation.document.abstract,
        )
        results.append(
            ExtractionEvaluationItem(
                annotation_id=annotation.id,
                pmid=annotation.document.pmid,
                attempt=attempt,
                metrics=metrics,
            )
        )

    metric_summary = mean_metrics(result.metrics for result in results)
    total = len(results)
    summary: dict[str, float | int] = {
        "items": total,
        "json_parse_rate": _rate(sum(result.attempt.json_parsed for result in results), total),
        "schema_validity_rate": _rate(sum(result.attempt.schema_valid for result in results), total),
        "mean_latency_ms": sum(result.attempt.latency_ms for result in results) / total if total else 0.0,
        **{f"mean_{key}": value for key, value in metric_summary.items()},
    }
    return ExtractionEvaluationReport(
        backend=backend.name,
        generated_at=datetime.now(timezone.utc),
        summary=summary,
        items=tuple(results),
    )


def format_extraction_report(report: ExtractionEvaluationReport) -> str:
    summary = report.summary
    return "\n".join(
        [
            "Evidence extraction evaluation",
            f"Backend: {report.backend}",
            f"Items: {summary['items']}",
            f"JSON parse rate: {summary['json_parse_rate']:.4f}",
            f"Schema validity rate: {summary['schema_validity_rate']:.4f}",
            f"Evidence status accuracy: {summary['mean_evidence_status_accuracy']:.4f}",
            f"Study design accuracy: {summary['mean_study_design_accuracy']:.4f}",
            f"Semantic field token F1: {summary['mean_semantic_field_token_f1']:.4f}",
            f"Outcome direction accuracy: {summary['mean_outcome_direction_accuracy']:.4f}",
            f"Evidence span token F1: {summary['mean_evidence_span_token_f1']:.4f}",
            f"Evidence span support rate: {summary['mean_evidence_span_support_rate']:.4f}",
            f"Mean latency: {summary['mean_latency_ms']:.2f} ms",
        ]
    )


def write_extraction_report(report: ExtractionEvaluationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
