from __future__ import annotations

from bioevidence.evaluation.metrics import (
    compute_answer_metrics,
    compute_citation_metrics,
    compute_metrics,
    compute_retrieval_metrics,
)


def test_retrieval_metrics_are_deterministic():
    metrics = compute_retrieval_metrics(["111", "222", "333"], ["333", "444"])

    assert metrics == {
        "hit_at_k": 1.0,
        "recall_at_k": 0.5,
        "mrr": 1 / 3,
    }


def test_citation_metrics_are_stable():
    metrics = compute_citation_metrics(["111", "222"], ["222", "333"])

    assert metrics == {
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }


def test_answer_metrics_cover_exact_match_and_token_overlap():
    metrics = compute_answer_metrics(
        "Corticosteroids reduce asthma exacerbations.",
        "Corticosteroids reduce asthma exacerbations.",
    )

    assert metrics == {
        "exact_match": 1.0,
        "token_overlap": 1.0,
    }


def test_compute_metrics_aggregates_text_scores():
    metrics = compute_metrics(
        ["Corticosteroids reduce asthma exacerbations.", "One more answer"],
        ["Corticosteroids reduce asthma exacerbations.", "Different answer"],
    )

    assert metrics["exact_match"] == 0.5
    assert 0.0 < metrics["token_overlap"] <= 1.0

