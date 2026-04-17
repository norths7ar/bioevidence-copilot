from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Iterable

from bioevidence.agent.workflow import WorkflowResult, run_rag_pipeline
from bioevidence.config import Settings, load_settings
from bioevidence.evaluation.dataset import EvaluationItem, load_dataset
from bioevidence.evaluation.metrics import (
    compute_answer_metrics,
    compute_citation_metrics,
    compute_retrieval_metrics,
)
from bioevidence.extraction.table import evidence_table_rows
from bioevidence.schemas.query import Query


PipelineFn = Callable[..., WorkflowResult]


@dataclass(frozen=True, slots=True)
class EvaluationItemResult:
    item: EvaluationItem
    predicted_pmids: tuple[str, ...]
    predicted_citations: tuple[str, ...]
    retrieval_metrics: dict[str, float]
    citation_metrics: dict[str, float]
    answer_metrics: dict[str, float | None]
    evidence_table: tuple[dict[str, object], ...]
    answer_text: str
    rewritten_query: str
    retrieval_source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "item": {
                "id": self.item.id,
                "query": self.item.query,
                "gold_pmids": list(self.item.gold_pmids),
                "reference_answer": self.item.reference_answer,
                "top_k": self.item.top_k,
            },
            "predicted_pmids": list(self.predicted_pmids),
            "predicted_citations": list(self.predicted_citations),
            "retrieval_metrics": self.retrieval_metrics,
            "citation_metrics": self.citation_metrics,
            "answer_metrics": self.answer_metrics,
            "evidence_table": list(self.evidence_table),
            "answer_text": self.answer_text,
            "rewritten_query": self.rewritten_query,
            "retrieval_source": self.retrieval_source,
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    dataset_path: Path
    generated_at: datetime
    summary: dict[str, float | int | None]
    items: tuple[EvaluationItemResult, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset_path": str(self.dataset_path),
            "generated_at": self.generated_at.isoformat(),
            "summary": self.summary,
            "items": [item.to_dict() for item in self.items],
        }


def run_evaluation(
    dataset_path: Path,
    *,
    pipeline: PipelineFn = run_rag_pipeline,
    data_dir: Path | None = None,
    settings: Settings | None = None,
) -> EvaluationReport:
    if settings is None and pipeline is run_rag_pipeline:
        settings = load_settings()
    items = load_dataset(dataset_path)
    results: list[EvaluationItemResult] = []

    for item in items:
        workflow_result = pipeline(
            Query(text=item.query, top_k=item.top_k),
            data_dir=data_dir,
            settings=settings,
        )
        results.append(_evaluate_item(item, workflow_result))

    return EvaluationReport(
        dataset_path=dataset_path,
        generated_at=datetime.now(timezone.utc),
        summary=_summarize(results),
        items=tuple(results),
    )


def format_report(report: EvaluationReport) -> str:
    def _format_metric(value: float | int | None) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, int):
            return str(value)
        return f"{value:.4f}"

    lines = [
        "Evaluation report",
        f"Dataset: {report.dataset_path}",
        f"Items: {_format_metric(report.summary.get('items', 0))}",
        f"Reference answers: {_format_metric(report.summary.get('reference_items', 0))}",
        "Retrieval:",
        f"  hit@k: {_format_metric(report.summary.get('mean_hit_at_k', 0.0))}",
        f"  recall@k: {_format_metric(report.summary.get('mean_recall_at_k', 0.0))}",
        f"  mrr: {_format_metric(report.summary.get('mean_mrr', 0.0))}",
        "Citations:",
        f"  precision: {_format_metric(report.summary.get('mean_citation_precision', 0.0))}",
        f"  recall: {_format_metric(report.summary.get('mean_citation_recall', 0.0))}",
        f"  f1: {_format_metric(report.summary.get('mean_citation_f1', 0.0))}",
        "Answers:",
        f"  exact_match: {_format_metric(report.summary.get('mean_answer_exact_match', None))}",
        f"  token_overlap: {_format_metric(report.summary.get('mean_answer_token_overlap', None))}",
    ]
    return "\n".join(lines)


def _evaluate_item(item: EvaluationItem, workflow_result: WorkflowResult) -> EvaluationItemResult:
    retrieved_candidates = workflow_result.retrieved_candidates[: item.top_k]
    predicted_pmids = tuple(candidate.document.pmid for candidate in retrieved_candidates)
    predicted_citations = tuple(workflow_result.answer.citations)
    retrieval_metrics = compute_retrieval_metrics(predicted_pmids, item.gold_pmids)
    citation_metrics = compute_citation_metrics(predicted_citations, item.gold_pmids)
    answer_metrics = _answer_metrics(workflow_result.answer.answer_text, item.reference_answer)

    return EvaluationItemResult(
        item=item,
        predicted_pmids=predicted_pmids,
        predicted_citations=predicted_citations,
        retrieval_metrics=retrieval_metrics,
        citation_metrics=citation_metrics,
        answer_metrics=answer_metrics,
        evidence_table=tuple(evidence_table_rows(workflow_result.evidence_records)),
        answer_text=workflow_result.answer.answer_text,
        rewritten_query=workflow_result.answer.rewritten_query or item.query,
        retrieval_source=workflow_result.source,
    )


def _answer_metrics(answer_text: str, reference_answer: str | None) -> dict[str, float | None]:
    if reference_answer is None:
        return {"exact_match": None, "token_overlap": None}
    metrics = compute_answer_metrics(answer_text, reference_answer)
    return {
        "exact_match": metrics["exact_match"],
        "token_overlap": metrics["token_overlap"],
    }


def _summarize(results: list[EvaluationItemResult]) -> dict[str, float | int | None]:
    summary: dict[str, float | int | None] = {
        "items": len(results),
        "reference_items": sum(1 for result in results if result.item.reference_answer is not None),
        "mean_hit_at_k": _mean(result.retrieval_metrics["hit_at_k"] for result in results),
        "mean_recall_at_k": _mean(result.retrieval_metrics["recall_at_k"] for result in results),
        "mean_mrr": _mean(result.retrieval_metrics["mrr"] for result in results),
        "mean_citation_precision": _mean(result.citation_metrics["precision"] for result in results),
        "mean_citation_recall": _mean(result.citation_metrics["recall"] for result in results),
        "mean_citation_f1": _mean(result.citation_metrics["f1"] for result in results),
        "mean_answer_exact_match": _mean(
            result.answer_metrics["exact_match"]
            for result in results
            if result.answer_metrics["exact_match"] is not None
        ),
        "mean_answer_token_overlap": _mean(
            result.answer_metrics["token_overlap"]
            for result in results
            if result.answer_metrics["token_overlap"] is not None
        ),
    }
    if summary["reference_items"] == 0:
        summary["mean_answer_exact_match"] = None
        summary["mean_answer_token_overlap"] = None
    return summary


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def write_report(report: EvaluationReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
