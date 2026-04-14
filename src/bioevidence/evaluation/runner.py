from __future__ import annotations

from pathlib import Path

from bioevidence.evaluation.dataset import load_dataset


def run_evaluation(dataset_path: Path) -> dict[str, float | int]:
    _items = load_dataset(dataset_path)
    return {"items": len(_items), "exact_match": 0.0}
