from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from bioevidence.schemas.model_evidence import ModelEvidenceExtraction, OutcomeEvidence, unsupported_evidence_spans


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_SEMANTIC_FIELDS = ("population_or_system", "intervention_or_exposure", "comparator")


def compute_extraction_metrics(
    predicted: ModelEvidenceExtraction | None,
    expected: ModelEvidenceExtraction,
    *,
    abstract: str,
) -> dict[str, float]:
    if predicted is None:
        return {
            "evidence_status_accuracy": 0.0,
            "study_design_accuracy": 0.0,
            "semantic_field_token_f1": 0.0,
            "outcome_name_token_f1": 0.0,
            "outcome_direction_accuracy": 0.0,
            "evidence_span_token_f1": 0.0,
            "evidence_span_support_rate": 0.0,
        }

    outcome_metrics = _outcome_metrics(predicted.outcomes, expected.outcomes)
    supported = len(predicted.outcomes) - len(unsupported_evidence_spans(predicted, abstract))
    support_rate = supported / len(predicted.outcomes) if predicted.outcomes else 1.0
    return {
        "evidence_status_accuracy": float(predicted.evidence_status == expected.evidence_status),
        "study_design_accuracy": float(predicted.study_design == expected.study_design),
        "semantic_field_token_f1": _mean(
            _text_f1(getattr(predicted, field), getattr(expected, field)) for field in _SEMANTIC_FIELDS
        ),
        **outcome_metrics,
        "evidence_span_support_rate": support_rate,
    }


def _outcome_metrics(
    predicted: tuple[OutcomeEvidence, ...],
    expected: tuple[OutcomeEvidence, ...],
) -> dict[str, float]:
    if not predicted and not expected:
        return {
            "outcome_name_token_f1": 1.0,
            "outcome_direction_accuracy": 1.0,
            "evidence_span_token_f1": 1.0,
        }
    if not predicted or not expected:
        return {
            "outcome_name_token_f1": 0.0,
            "outcome_direction_accuracy": 0.0,
            "evidence_span_token_f1": 0.0,
        }

    remaining = list(expected)
    name_scores: list[float] = []
    direction_scores: list[float] = []
    span_scores: list[float] = []
    for predicted_outcome in predicted:
        best_index = max(
            range(len(remaining)),
            key=lambda index: _text_f1(predicted_outcome.name, remaining[index].name),
            default=-1,
        )
        if best_index < 0:
            name_scores.append(0.0)
            direction_scores.append(0.0)
            span_scores.append(0.0)
            continue
        expected_outcome = remaining.pop(best_index)
        name_scores.append(_text_f1(predicted_outcome.name, expected_outcome.name))
        direction_scores.append(float(predicted_outcome.direction == expected_outcome.direction))
        span_scores.append(_text_f1(predicted_outcome.evidence_span, expected_outcome.evidence_span))
    missing = len(remaining)
    denominator = len(predicted) + missing
    return {
        "outcome_name_token_f1": sum(name_scores) / denominator,
        "outcome_direction_accuracy": sum(direction_scores) / denominator,
        "evidence_span_token_f1": sum(span_scores) / denominator,
    }


def _text_f1(predicted: str | None, expected: str | None) -> float:
    if predicted is None and expected is None:
        return 1.0
    predicted_tokens = Counter(_TOKEN_PATTERN.findall((predicted or "").casefold()))
    expected_tokens = Counter(_TOKEN_PATTERN.findall((expected or "").casefold()))
    if not predicted_tokens or not expected_tokens:
        return 0.0
    overlap = sum((predicted_tokens & expected_tokens).values())
    precision = overlap / sum(predicted_tokens.values())
    recall = overlap / sum(expected_tokens.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def mean_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {}
    return {key: _mean(row[key] for row in rows) for key in rows[0]}


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
