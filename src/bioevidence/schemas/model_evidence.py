from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class EvidenceStatus(StrEnum):
    DIRECT = "direct"
    INDIRECT = "indirect"
    NONE = "none"
    UNCLEAR = "unclear"


class StudyDesign(StrEnum):
    RANDOMIZED_CONTROLLED_TRIAL = "randomized_controlled_trial"
    NON_RANDOMIZED_INTERVENTIONAL = "non_randomized_interventional"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CROSS_SECTIONAL = "cross_sectional"
    CASE_REPORT_OR_SERIES = "case_report_or_series"
    SYSTEMATIC_REVIEW_OR_META_ANALYSIS = "systematic_review_or_meta_analysis"
    NARRATIVE_REVIEW = "narrative_review"
    STUDY_PROTOCOL = "study_protocol"
    PRECLINICAL_IN_VIVO = "preclinical_in_vivo"
    IN_VITRO = "in_vitro"
    OTHER = "other"
    NOT_REPORTED = "not_reported"


class OutcomeDirection(StrEnum):
    INCREASED = "increased"
    DECREASED = "decreased"
    NO_CLEAR_DIFFERENCE = "no_clear_difference"
    MIXED = "mixed"
    ASSOCIATION_ONLY = "association_only"
    NOT_REPORTED = "not_reported"


class OutcomeEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: NonEmptyText
    direction: OutcomeDirection
    result_text: NonEmptyText | None
    evidence_span: NonEmptyText


class ModelEvidenceExtraction(BaseModel):
    """Query-focused fields predicted from one PubMed title and abstract."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        json_schema_extra={
            "$id": "https://bioevidence.local/schemas/model-evidence-extraction-v1.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    evidence_status: EvidenceStatus
    study_design: StudyDesign
    population_or_system: NonEmptyText | None
    intervention_or_exposure: NonEmptyText | None
    comparator: NonEmptyText | None
    outcomes: tuple[OutcomeEvidence, ...] = Field(max_length=8)
    evidence_summary: NonEmptyText | None

    @model_validator(mode="after")
    def validate_status_consistency(self) -> Self:
        content_fields = (
            self.population_or_system,
            self.intervention_or_exposure,
            self.comparator,
            self.evidence_summary,
        )
        if self.evidence_status is EvidenceStatus.NONE and (any(content_fields) or self.outcomes):
            raise ValueError("none evidence must not contain extracted evidence fields")
        if self.evidence_status in {EvidenceStatus.DIRECT, EvidenceStatus.INDIRECT} and self.evidence_summary is None:
            raise ValueError("direct and indirect evidence require an evidence_summary")
        if self.evidence_status is EvidenceStatus.DIRECT and not self.outcomes:
            raise ValueError("direct evidence requires at least one grounded outcome")
        return self


def unsupported_evidence_spans(extraction: ModelEvidenceExtraction, abstract: str) -> tuple[str, ...]:
    """Return outcome spans that are not verbatim substrings of the source abstract."""

    return tuple(outcome.evidence_span for outcome in extraction.outcomes if outcome.evidence_span not in abstract)
