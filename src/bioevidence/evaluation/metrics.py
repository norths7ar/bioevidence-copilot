from __future__ import annotations

from collections import Counter
import re
from collections.abc import Sequence


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    exact_match_scores = []
    overlap_scores = []
    for prediction, reference in zip(predictions, references, strict=False):
        exact_match_scores.append(1.0 if normalized_exact_match(prediction, reference) else 0.0)
        overlap_scores.append(token_overlap_f1(prediction, reference))
    return {
        "exact_match": _mean(exact_match_scores),
        "token_overlap": _mean(overlap_scores),
    }


def compute_retrieval_metrics(predicted_pmids: Sequence[str], gold_pmids: Sequence[str]) -> dict[str, float]:
    predicted = _unique_strings(predicted_pmids)
    gold = _unique_strings(gold_pmids)
    if not gold:
        raise ValueError("gold_pmids must not be empty")

    gold_set = set(gold)
    hits = [pmid for pmid in predicted if pmid in gold_set]
    first_hit_rank = next((index + 1 for index, pmid in enumerate(predicted) if pmid in gold_set), None)
    return {
        "hit_at_k": 1.0 if hits else 0.0,
        "recall_at_k": len(set(hits)) / len(gold_set),
        "mrr": 1.0 / first_hit_rank if first_hit_rank is not None else 0.0,
    }


def compute_citation_metrics(predicted_citations: Sequence[str], gold_pmids: Sequence[str]) -> dict[str, float]:
    predicted = set(_unique_strings(predicted_citations))
    gold = set(_unique_strings(gold_pmids))
    if not gold:
        raise ValueError("gold_pmids must not be empty")
    overlap = predicted & gold
    precision = len(overlap) / len(predicted) if predicted else 0.0
    recall = len(overlap) / len(gold)
    f1 = _f1(precision, recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_answer_metrics(prediction: str, reference: str) -> dict[str, float]:
    return {
        "exact_match": 1.0 if normalized_exact_match(prediction, reference) else 0.0,
        "token_overlap": token_overlap_f1(prediction, reference),
    }


def normalized_exact_match(prediction: str, reference: str) -> bool:
    return _normalize_for_match(prediction) == _normalize_for_match(reference)


def token_overlap_f1(prediction: str, reference: str) -> float:
    prediction_tokens = _tokenize(prediction)
    reference_tokens = _tokenize(reference)
    if not prediction_tokens and not reference_tokens:
        return 1.0
    if not prediction_tokens or not reference_tokens:
        return 0.0

    predicted_counts = Counter(prediction_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum(min(predicted_counts[token], reference_counts[token]) for token in predicted_counts.keys() & reference_counts.keys())
    precision = overlap / sum(predicted_counts.values())
    recall = overlap / sum(reference_counts.values())
    return _f1(precision, recall)


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _normalize_for_match(text: str) -> str:
    return " ".join(_tokenize(text))


def _unique_strings(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _mean(values: Sequence[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
