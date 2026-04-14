from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    query: str
    expected_answer: str = ""


def load_dataset(path: Path) -> list[EvaluationItem]:
    _ = path
    return []
