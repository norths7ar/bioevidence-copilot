from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from unsloth import FastLanguageModel

from bioevidence.extraction.model_backend import build_extraction_messages
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction, unsupported_evidence_spans


DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def main() -> int:
    args = _parse_args()
    annotation = _find_jsonl_record(args.dataset, "id", args.annotation_id)
    document = Document(**_find_jsonl_record(args.corpus, "pmid", annotation["pmid"]))

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

    messages = build_extraction_messages(annotation["query"], document)
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to("cuda")

    generation_started = perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generation_seconds = perf_counter() - generation_started
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    parsed_payload = _parse_json_object(raw_output)
    extraction = ModelEvidenceExtraction.model_validate(parsed_payload)
    unsupported_spans = unsupported_evidence_spans(extraction, document.abstract)
    report = {
        "model": args.model,
        "annotation_id": args.annotation_id,
        "pmid": document.pmid,
        "load_seconds": round(load_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "generated_tokens": int(generated_ids.shape[0]),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "json_parsed": True,
        "schema_valid": True,
        "grounded": not unsupported_spans,
        "unsupported_spans": list(unsupported_spans),
        "prediction": extraction.model_dump(mode="json"),
        "raw_output": raw_output,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not unsupported_spans else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local structured extraction smoke test.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--annotation-id", default="diabetes-42185896")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
    )
    parser.add_argument("--corpus", type=Path, default=Path("data/corpora/demo/processed/demo.documents.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training/evidence_extraction/smoke_test.json"),
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args()


def _find_jsonl_record(path: Path, key: str, value: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        record = json.loads(line)
        if record.get(key) == value:
            return record
    raise ValueError(f"No record with {key}={value!r} in {path}")


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
    raise ValueError(f"Model did not return a JSON object:\n{content}")


if __name__ == "__main__":
    raise SystemExit(main())
