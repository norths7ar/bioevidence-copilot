from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationItem:
    id: str
    query: str
    gold_pmids: tuple[str, ...]
    reference_answer: str | None = None
    top_k: int = 10


def load_dataset(path: Path) -> list[EvaluationItem]:
    if not path.exists():
        raise FileNotFoundError(path)

    items: list[EvaluationItem] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        items.append(_parse_item(payload, line_number))
    return items


def _parse_item(payload: Any, line_number: int) -> EvaluationItem:
    if not isinstance(payload, dict):
        raise ValueError(f"Line {line_number}: expected a JSON object")

    item_id = _require_str(payload, "id", line_number)
    query = _require_str(payload, "query", line_number)
    gold_pmids = _parse_pmids(payload, line_number)
    reference_answer = payload.get("reference_answer")
    if reference_answer is not None and (
        not isinstance(reference_answer, str) or not reference_answer.strip()
    ):
        raise ValueError(f"Line {line_number}: reference_answer must be a non-empty string if provided")

    top_k_value = payload.get("top_k", 10)
    if isinstance(top_k_value, bool) or not isinstance(top_k_value, int) or top_k_value <= 0:
        raise ValueError(f"Line {line_number}: top_k must be a positive integer")

    return EvaluationItem(
        id=item_id,
        query=query,
        gold_pmids=gold_pmids,
        reference_answer=reference_answer.strip() if reference_answer is not None else None,
        top_k=top_k_value,
    )


def _require_str(payload: dict[str, Any], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {key} must be a non-empty string")
    return value.strip()


def _parse_pmids(payload: dict[str, Any], line_number: int) -> tuple[str, ...]:
    raw_pmids = payload.get("gold_pmids")
    if raw_pmids is None:
        raw_pmids = payload.get("gold_citations")
    if raw_pmids is None:
        raise ValueError(f"Line {line_number}: gold_pmids (or gold_citations) is required")

    if isinstance(raw_pmids, str):
        values = [raw_pmids]
    elif isinstance(raw_pmids, (list, tuple)):
        values = list(raw_pmids)
    else:
        raise ValueError(f"Line {line_number}: gold_pmids must be a string or list of strings")

    pmids = tuple(str(pmid).strip() for pmid in values if str(pmid).strip())
    if not pmids:
        raise ValueError(f"Line {line_number}: gold_pmids cannot be empty")
    return pmids
