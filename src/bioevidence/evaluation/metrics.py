from __future__ import annotations


def compute_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    _ = (predictions, references)
    return {"exact_match": 0.0}
