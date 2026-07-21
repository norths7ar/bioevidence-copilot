from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction, unsupported_evidence_spans
from bioevidence.schemas.model_evidence import EvidenceStatus, OutcomeDirection, OutcomeEvidence, StudyDesign


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class ExtractionBackend(Protocol):
    name: str

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction: ...


class ExtractionBackendError(RuntimeError):
    def __init__(self, message: str, *, kind: str, raw_output: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.raw_output = raw_output


@dataclass(frozen=True, slots=True)
class ExtractionAttempt:
    extraction: ModelEvidenceExtraction | None
    latency_ms: float
    error_kind: str | None = None
    error_message: str | None = None
    raw_output: str = ""

    @property
    def json_parsed(self) -> bool:
        return self.error_kind not in {"request", "empty", "json"}

    @property
    def schema_valid(self) -> bool:
        return self.extraction is not None or self.error_kind == "grounding"


def run_extraction_attempt(
    backend: ExtractionBackend,
    query: str,
    document: Document,
) -> ExtractionAttempt:
    started_at = perf_counter()
    try:
        extraction = backend.extract(query, document)
    except ExtractionBackendError as exc:
        return ExtractionAttempt(
            extraction=None,
            latency_ms=(perf_counter() - started_at) * 1000,
            error_kind=exc.kind,
            error_message=str(exc),
            raw_output=exc.raw_output,
        )
    return ExtractionAttempt(extraction=extraction, latency_ms=(perf_counter() - started_at) * 1000)


class PromptedExtractionBackend:
    name = "prompted"

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        max_output_tokens: int = 2048,
        temperature: float = 0.0,
        client: OpenAI | None = None,
        completion: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be configured")
        if completion is None and (not api_key.strip() or not base_url.strip()):
            raise ValueError("api_key and base_url must be configured")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self._client = client or (OpenAI(api_key=api_key, base_url=base_url) if completion is None else None)
        self._completion = completion

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction:
        messages = build_extraction_messages(query, document)
        raw_output = self._complete(messages)
        payload = _parse_json_object(raw_output)
        try:
            extraction = ModelEvidenceExtraction.model_validate(payload)
        except ValidationError as exc:
            raise ExtractionBackendError(
                "Model output failed the extraction schema",
                kind="schema",
                raw_output=raw_output,
            ) from exc
        if unsupported_evidence_spans(extraction, document.abstract):
            raise ExtractionBackendError(
                "Model output contains evidence spans not copied from the abstract",
                kind="grounding",
                raw_output=raw_output,
            )
        return extraction

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self._completion is not None:
            return self._completion(messages).strip()
        assert self._client is not None
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                max_tokens=self.max_output_tokens,
                temperature=self.temperature,
            )
        except OpenAIError as exc:
            raise ExtractionBackendError("Extraction model request failed", kind="request") from exc
        return (response.choices[0].message.content or "").strip()


class RuleBasedExtractionBackend:
    """Small inspectable baseline for the model extraction contract."""

    name = "rules"

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction:
        text = f"{document.title} {document.abstract}".casefold()
        study_design = _infer_study_design(text)
        query_terms = {token for token in _tokens(query) if len(token) >= 4 and token not in _QUERY_STOPWORDS}
        matched_terms = {token for token in query_terms if token in text}
        if not matched_terms or study_design is StudyDesign.STUDY_PROTOCOL:
            return ModelEvidenceExtraction(
                evidence_status=EvidenceStatus.NONE,
                study_design=study_design,
                population_or_system=None,
                intervention_or_exposure=None,
                comparator=None,
                outcomes=(),
                evidence_summary=None,
            )

        result_span = _result_sentence(document.abstract)
        is_direct = bool(result_span) and len(matched_terms) >= min(2, len(query_terms))
        if is_direct:
            assert result_span is not None
            return ModelEvidenceExtraction(
                evidence_status=EvidenceStatus.DIRECT,
                study_design=study_design,
                population_or_system=None,
                intervention_or_exposure=None,
                comparator=None,
                outcomes=(
                    OutcomeEvidence(
                        name="reported query outcome",
                        direction=_infer_direction(result_span),
                        result_text=result_span,
                        evidence_span=result_span,
                    ),
                ),
                evidence_summary=result_span,
            )
        return ModelEvidenceExtraction(
            evidence_status=EvidenceStatus.INDIRECT,
            study_design=study_design,
            population_or_system=None,
            intervention_or_exposure=None,
            comparator=None,
            outcomes=(),
            evidence_summary=_first_sentence(document.abstract) or document.title,
        )


def build_extraction_messages(query: str, document: Document) -> list[dict[str, str]]:
    schema = json.dumps(ModelEvidenceExtraction.model_json_schema(), ensure_ascii=False, separators=(",", ":"))
    system_prompt = (
        "You extract query-focused biomedical evidence from one PubMed title and abstract. "
        "Return exactly one JSON object and no commentary. Follow the supplied JSON Schema exactly. "
        "Use null when the abstract does not support a nullable field. Every evidence_span must be copied "
        "verbatim from ABSTRACT. Do not use outside knowledge. If evidence_status is none, return null for "
        "all nullable evidence fields and an empty outcomes array."
    )
    user_prompt = (
        f"JSON_SCHEMA:\n{schema}\n\n"
        f"QUERY:\n{query.strip()}\n\n"
        f"TITLE:\n{document.title.strip()}\n\n"
        f"ABSTRACT:\n{document.abstract.strip()}"
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


def _parse_json_object(content: str) -> dict[str, Any]:
    if not content:
        raise ExtractionBackendError("Model returned empty content", kind="empty")
    candidates = [content]
    match = _JSON_FENCE_PATTERN.search(content)
    if match:
        candidates.insert(0, match.group(1))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ExtractionBackendError("Model did not return a JSON object", kind="json", raw_output=content)


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_QUERY_STOPWORDS = {
    "does",
    "have",
    "what",
    "with",
    "from",
    "this",
    "that",
    "evidence",
    "effect",
}
_RESULT_CUES = ("significant", "increased", "decreased", "reduced", "associated", "difference", "improved")


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(_TOKEN_PATTERN.findall(text.casefold()))


def _sentences(text: str) -> tuple[str, ...]:
    return tuple(sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip())


def _first_sentence(text: str) -> str | None:
    sentences = _sentences(text)
    return sentences[0] if sentences else None


def _result_sentence(text: str) -> str | None:
    return next((sentence for sentence in _sentences(text) if any(cue in sentence.casefold() for cue in _RESULT_CUES)), None)


def _infer_direction(text: str) -> OutcomeDirection:
    lowered = text.casefold()
    if "no significant" in lowered or "no difference" in lowered:
        return OutcomeDirection.NO_CLEAR_DIFFERENCE
    if any(term in lowered for term in ("decreased", "reduced", "lower")):
        return OutcomeDirection.DECREASED
    if any(term in lowered for term in ("increased", "improved", "higher")):
        return OutcomeDirection.INCREASED
    if "associated" in lowered:
        return OutcomeDirection.ASSOCIATION_ONLY
    return OutcomeDirection.NOT_REPORTED


def _infer_study_design(text: str) -> StudyDesign:
    if "study protocol" in text or "trial protocol" in text:
        return StudyDesign.STUDY_PROTOCOL
    if "meta-analysis" in text or "systematic review" in text:
        return StudyDesign.SYSTEMATIC_REVIEW_OR_META_ANALYSIS
    if "randomized" in text or "randomised" in text:
        return StudyDesign.RANDOMIZED_CONTROLLED_TRIAL
    if "case-control" in text:
        return StudyDesign.CASE_CONTROL
    if "cross-sectional" in text:
        return StudyDesign.CROSS_SECTIONAL
    if "cohort" in text:
        return StudyDesign.COHORT
    if "in vitro" in text:
        return StudyDesign.IN_VITRO
    if any(term in text for term in ("mice", "mouse model", "rats", "murine")):
        return StudyDesign.PRECLINICAL_IN_VIVO
    if "review" in text:
        return StudyDesign.NARRATIVE_REVIEW
    return StudyDesign.NOT_REPORTED
