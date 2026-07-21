from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from pydantic import ValidationError
from unsloth import FastLanguageModel

from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.evaluation.extraction_metrics import compute_extraction_metrics, mean_metrics
from bioevidence.extraction.model_backend import build_extraction_messages
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction, unsupported_evidence_spans


DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def main() -> int:
    args = _parse_args()
    annotations = load_extraction_annotations(args.dataset, _load_documents(args.corpus))
    if args.limit is not None:
        annotations = annotations[: args.limit]

    torch.cuda.reset_peak_memory_stats()
    load_started = perf_counter()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    FastLanguageModel.for_inference(model)
    load_seconds = perf_counter() - load_started

    results: list[dict[str, Any]] = []
    for index, annotation in enumerate(annotations, start=1):
        result = _evaluate_annotation(
            model,
            tokenizer,
            annotation,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(result)
        print(
            f"[{index}/{len(annotations)}] {annotation.id} "
            f"json={result['json_parsed']} schema={result['schema_valid']} "
            f"grounded={result['grounded']} latency={result['generation_seconds']:.2f}s",
            flush=True,
        )

    report = {
        "model": args.model_label or args.model,
        "model_source": args.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "load_seconds": round(load_seconds, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "summary": _summarize(results),
        "items": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Report: {args.output}", flush=True)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a local Unsloth extraction model.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-label", default="")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corpora/demo/processed/demo.documents.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/extraction_qwen3_4b_prompted.json"),
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _evaluate_annotation(model: Any, tokenizer: Any, annotation: Any, *, max_new_tokens: int) -> dict[str, Any]:
    messages = build_extraction_messages(annotation.query, annotation.document)
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to("cuda")
    started = perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generation_seconds = perf_counter() - started
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    prediction: ModelEvidenceExtraction | None = None
    error_kind: str | None = None
    error_message: str | None = None
    json_parsed = False
    grounded = False
    unsupported_spans: tuple[str, ...] = ()
    try:
        payload = _parse_json_object(raw_output)
        json_parsed = True
        prediction = ModelEvidenceExtraction.model_validate(payload)
        unsupported_spans = unsupported_evidence_spans(prediction, annotation.document.abstract)
        grounded = not unsupported_spans
    except json.JSONDecodeError as exc:
        error_kind = "json"
        error_message = str(exc)
    except ValidationError as exc:
        error_kind = "schema"
        error_message = str(exc)

    metrics = compute_extraction_metrics(prediction, annotation.extraction, abstract=annotation.document.abstract)
    return {
        "annotation_id": annotation.id,
        "pmid": annotation.document.pmid,
        "generation_seconds": round(generation_seconds, 3),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(generated_ids.shape[0]),
        "json_parsed": json_parsed,
        "schema_valid": prediction is not None,
        "grounded": grounded,
        "unsupported_spans": list(unsupported_spans),
        "error_kind": error_kind,
        "error_message": error_message,
        "prediction": prediction.model_dump(mode="json") if prediction else None,
        "raw_output": raw_output,
        "metrics": metrics,
    }


def _load_documents(path: Path) -> list[Document]:
    return [Document(**record) for record in _iter_jsonl(path)]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            yield payload


def _parse_json_object(content: str) -> dict[str, Any]:
    candidates = [content]
    match = JSON_FENCE_PATTERN.search(content)
    if match:
        candidates.insert(0, match.group(1))
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise json.JSONDecodeError("Model did not return a JSON object", content, 0)


def _summarize(results: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(results)
    metric_summary = mean_metrics(result["metrics"] for result in results)
    return {
        "items": total,
        "json_parse_rate": _rate(sum(bool(result["json_parsed"]) for result in results), total),
        "schema_validity_rate": _rate(sum(bool(result["schema_valid"]) for result in results), total),
        "grounding_rate": _rate(sum(bool(result["grounded"]) for result in results), total),
        "mean_generation_seconds": (
            sum(float(result["generation_seconds"]) for result in results) / total if total else 0.0
        ),
        **{f"mean_{key}": value for key, value in metric_summary.items()},
    }


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
