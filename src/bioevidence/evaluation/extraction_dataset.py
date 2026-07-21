from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction, unsupported_evidence_spans


class AnnotationStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"


@dataclass(frozen=True, slots=True)
class ExtractionAnnotation:
    id: str
    query: str
    document: Document
    extraction: ModelEvidenceExtraction
    annotation_status: AnnotationStatus


def load_extraction_annotations(
    path: Path,
    documents: Sequence[Document],
) -> list[ExtractionAnnotation]:
    if not path.exists():
        raise FileNotFoundError(path)

    documents_by_pmid = {document.pmid: document for document in documents}
    annotations: list[ExtractionAnnotation] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        payload = json.loads(line)
        annotation = _parse_annotation(payload, line_number, documents_by_pmid)
        if annotation.id in seen_ids:
            raise ValueError(f"Line {line_number}: duplicate annotation id {annotation.id!r}")
        seen_ids.add(annotation.id)
        annotations.append(annotation)
    return annotations


def _parse_annotation(
    payload: Any,
    line_number: int,
    documents_by_pmid: dict[str, Document],
) -> ExtractionAnnotation:
    if not isinstance(payload, dict):
        raise ValueError(f"Line {line_number}: expected a JSON object")

    annotation_id = _require_non_empty_str(payload, "id", line_number)
    query = _require_non_empty_str(payload, "query", line_number)
    pmid = _require_non_empty_str(payload, "pmid", line_number)
    document = documents_by_pmid.get(pmid)
    if document is None:
        raise ValueError(f"Line {line_number}: PMID {pmid!r} is not present in the supplied corpus")

    raw_annotation_status = payload.get("annotation_status")
    if not isinstance(raw_annotation_status, str):
        raise ValueError(f"Line {line_number}: annotation_status must be 'draft' or 'reviewed'")
    try:
        annotation_status = AnnotationStatus(raw_annotation_status)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: annotation_status must be 'draft' or 'reviewed'") from exc

    try:
        extraction = ModelEvidenceExtraction.model_validate(payload.get("extraction"))
    except ValidationError as exc:
        raise ValueError(f"Line {line_number}: invalid extraction payload: {exc}") from exc

    unsupported_spans = unsupported_evidence_spans(extraction, document.abstract)
    if unsupported_spans:
        raise ValueError(f"Line {line_number}: evidence_span must be copied verbatim from PMID {pmid}")

    return ExtractionAnnotation(
        id=annotation_id,
        query=query,
        document=document,
        extraction=extraction,
        annotation_status=annotation_status,
    )


def _require_non_empty_str(payload: dict[str, Any], key: str, line_number: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: {key} must be a non-empty string")
    return value.strip()
