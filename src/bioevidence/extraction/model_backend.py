from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction, unsupported_evidence_spans
from bioevidence.schemas.model_evidence import EvidenceStatus, OutcomeDirection, OutcomeEvidence, StudyDesign

if TYPE_CHECKING:
    from bioevidence.config import Settings


_JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
LOGGER = logging.getLogger(__name__)


class ExtractionBackend(Protocol):
    name: str

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction: ...


class ExtractionBackendError(RuntimeError):
    def __init__(self, message: str, *, kind: str, raw_output: str = "", details: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.raw_output = raw_output
        self.details = details


@dataclass(frozen=True, slots=True)
class ExtractionAttempt:
    extraction: ModelEvidenceExtraction | None
    latency_ms: float
    error_kind: str | None = None
    error_message: str | None = None
    error_details: str = ""
    raw_output: str = ""

    @property
    def json_parsed(self) -> bool:
        return self.error_kind not in {"unavailable", "request", "empty", "json"}

    @property
    def schema_valid(self) -> bool:
        return self.extraction is not None or self.error_kind == "grounding"


@dataclass(frozen=True, slots=True)
class ExtractionResolution:
    extraction: ModelEvidenceExtraction
    attempted_backend: str
    used_backend: str
    fallback_reason: str | None = None
    failed_raw_output: str = ""
    failure_details: str = ""


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
            error_details=exc.details,
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
        return _validate_model_output(raw_output, document)

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


class LocalAdapterExtractionBackend:
    """Lazy local inference adapter without adding training packages to product dependencies."""

    name = "local_adapter"

    def __init__(
        self,
        *,
        adapter_path: Path,
        max_seq_length: int = 4096,
        max_output_tokens: int = 1024,
        completion: Callable[[list[dict[str, str]]], str] | None = None,
    ) -> None:
        if max_seq_length <= 0 or max_output_tokens <= 0:
            raise ValueError("sequence and output token limits must be positive")
        self.adapter_path = adapter_path
        self.max_seq_length = max_seq_length
        self.max_output_tokens = max_output_tokens
        self._completion = completion
        self._runtime: tuple[Any, Any, Any] | None = None

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction:
        raw_output = self._complete(build_extraction_messages(query, document))
        return _validate_model_output(raw_output, document)

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self._completion is not None:
            return self._completion(messages).strip()
        model, tokenizer, torch = self._load_runtime()
        rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(rendered_prompt, return_tensors="pt").to("cuda")
        try:
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.max_output_tokens,
                    do_sample=False,
                    use_cache=True,
                )
        except Exception as exc:
            raise ExtractionBackendError("Local adapter inference failed", kind="request") from exc
        generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    def _load_runtime(self) -> tuple[Any, Any, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self.adapter_path.is_dir():
            raise ExtractionBackendError(
                f"Local adapter directory does not exist: {self.adapter_path}",
                kind="unavailable",
            )
        import_started = perf_counter()
        LOGGER.info("local_adapter_runtime_import_started")
        try:
            import torch
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise ExtractionBackendError(
                "Local adapter inference requires the separate training environment",
                kind="unavailable",
            ) from exc
        LOGGER.info(
            "local_adapter_runtime_import_completed duration_ms=%.1f",
            (perf_counter() - import_started) * 1000,
        )
        load_started = perf_counter()
        LOGGER.info("local_adapter_model_load_started")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=str(self.adapter_path),
                max_seq_length=self.max_seq_length,
                load_in_4bit=True,
                full_finetuning=False,
            )
            FastLanguageModel.for_inference(model)
        except Exception as exc:
            raise ExtractionBackendError("Local adapter could not be loaded", kind="unavailable") from exc
        LOGGER.info(
            "local_adapter_model_load_completed duration_ms=%.1f",
            (perf_counter() - load_started) * 1000,
        )
        self._runtime = (model, tokenizer, torch)
        return self._runtime


class FallbackExtractionBackend:
    """Use a deterministic backend when an optional model backend is unavailable."""

    def __init__(self, primary: ExtractionBackend, fallback: ExtractionBackend) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+{fallback.name}_fallback"
        self._primary_unavailable = False

    def extract(self, query: str, document: Document) -> ModelEvidenceExtraction:
        return self.resolve(query, document).extraction

    def resolve(self, query: str, document: Document) -> ExtractionResolution:
        if self._primary_unavailable:
            return ExtractionResolution(
                extraction=self.fallback.extract(query, document),
                attempted_backend=self.primary.name,
                used_backend=self.fallback.name,
                fallback_reason="unavailable",
            )
        try:
            extraction = self.primary.extract(query, document)
        except ExtractionBackendError as exc:
            LOGGER.warning("extraction_backend_fallback backend=%s reason=%s", self.primary.name, exc.kind)
            self._primary_unavailable = exc.kind == "unavailable"
            return ExtractionResolution(
                extraction=self.fallback.extract(query, document),
                attempted_backend=self.primary.name,
                used_backend=self.fallback.name,
                fallback_reason=exc.kind,
                failed_raw_output=exc.raw_output,
                failure_details=exc.details,
            )
        return ExtractionResolution(
            extraction=extraction,
            attempted_backend=self.primary.name,
            used_backend=self.primary.name,
        )


def resolve_extraction(
    backend: ExtractionBackend,
    query: str,
    document: Document,
) -> ExtractionResolution:
    if isinstance(backend, FallbackExtractionBackend):
        return backend.resolve(query, document)
    return ExtractionResolution(
        extraction=backend.extract(query, document),
        attempted_backend=backend.name,
        used_backend=backend.name,
    )


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


def create_product_extraction_backend(settings: Settings) -> ExtractionBackend | None:
    """Build the optional semantic extractor used by product workflows."""

    backend_name = settings.extraction_backend.strip().casefold()
    if backend_name in {"", "legacy"}:
        return None
    fallback = RuleBasedExtractionBackend()
    if backend_name == "rules":
        return fallback
    if backend_name == "prompted":
        if not (settings.extraction_api_key and settings.extraction_base_url and settings.extraction_model):
            LOGGER.warning("extraction_backend_fallback backend=prompted reason=unconfigured")
            return fallback
        primary: ExtractionBackend = PromptedExtractionBackend(
            api_key=settings.extraction_api_key,
            base_url=settings.extraction_base_url,
            model=settings.extraction_model,
            max_output_tokens=settings.extraction_max_output_tokens,
        )
        return FallbackExtractionBackend(primary, fallback)
    if backend_name == "local":
        if settings.extraction_adapter_path is None:
            LOGGER.warning("extraction_backend_fallback backend=local_adapter reason=unconfigured")
            return fallback
        primary = LocalAdapterExtractionBackend(
            adapter_path=settings.extraction_adapter_path,
            max_seq_length=settings.extraction_max_seq_length,
            max_output_tokens=settings.extraction_max_output_tokens,
        )
        return FallbackExtractionBackend(primary, fallback)
    raise ValueError(f"Unsupported extraction backend: {settings.extraction_backend}")


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


def _validate_model_output(raw_output: str, document: Document) -> ModelEvidenceExtraction:
    payload = _parse_json_object(raw_output)
    try:
        extraction = ModelEvidenceExtraction.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionBackendError(
            "Model output failed the extraction schema",
            kind="schema",
            raw_output=raw_output,
            details=json.dumps(
                exc.errors(include_url=False, include_input=False),
                ensure_ascii=False,
                default=str,
            ),
        ) from exc
    unsupported_spans = unsupported_evidence_spans(extraction, document.abstract)
    if unsupported_spans:
        raise ExtractionBackendError(
            "Model output contains evidence spans not copied from the abstract",
            kind="grounding",
            raw_output=raw_output,
            details=json.dumps({"unsupported_evidence_spans": unsupported_spans}, ensure_ascii=False),
        )
    return extraction


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
