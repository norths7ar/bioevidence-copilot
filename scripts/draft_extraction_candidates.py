from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from bioevidence.extraction.model_backend import PromptedExtractionBackend, run_extraction_attempt
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.schemas.document import Document


@dataclass(frozen=True, slots=True)
class DraftConfig:
    api_key: str
    base_url: str
    model: str
    max_output_tokens: int


def main() -> int:
    args = _parse_args()
    candidates = _load_jsonl(args.candidates)
    if args.limit is not None:
        candidates = candidates[: args.limit]
    documents = {document.pmid: document for document in load_local_documents(args.data_dir)}
    _validate_candidates(candidates, documents)
    config = _load_config(require_credentials=not args.dry_run)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "candidates": len(candidates),
                    "unique_pmids": len({candidate["pmid"] for candidate in candidates}),
                    "model": config.model or "unconfigured",
                    "base_url_configured": bool(config.base_url),
                    "api_key_configured": bool(config.api_key),
                    "max_workers": args.max_workers,
                },
                indent=2,
            )
        )
        return 0

    backend = PromptedExtractionBackend(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
        max_output_tokens=config.max_output_tokens,
        temperature=0.0,
    )
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        results = list(
            executor.map(
                lambda candidate: _draft_candidate(candidate, documents[candidate["pmid"]], backend),
                candidates,
            )
        )

    successes = [result["annotation"] for result in results if result["annotation"] is not None]
    failures = [result for result in results if result["annotation"] is None]
    _write_jsonl(args.output, successes)
    _write_jsonl(args.failures_output, failures)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_candidates": args.candidates.as_posix(),
        "model": config.model,
        "base_url": config.base_url,
        "requested": len(candidates),
        "drafted": len(successes),
        "failed": len(failures),
        "failure_kinds": dict(sorted(Counter(result["error_kind"] or "unknown" for result in failures).items())),
        "output": args.output.as_posix(),
        "failures_output": args.failures_output.as_posix(),
    }
    _write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if successes else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draft schema-valid annotations with an OpenAI-compatible model.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/evaluations/evidence_extraction/expansion_candidates.v1.jsonl"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/corpora/demo"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/annotation_drafts/extraction_expansion.v1.jsonl"),
    )
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=Path("artifacts/annotation_drafts/extraction_expansion.v1.failures.jsonl"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/annotation_drafts/extraction_expansion.v1.report.json"),
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_workers <= 0:
        parser.error("--max-workers must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def _load_config(*, require_credentials: bool) -> DraftConfig:
    load_dotenv()
    api_key = os.getenv("EXTRACTION_API_KEY") or os.getenv("AGENT_API_KEY", "")
    base_url = os.getenv("EXTRACTION_BASE_URL") or os.getenv("AGENT_BASE_URL", "")
    model = os.getenv("EXTRACTION_MODEL") or os.getenv("AGENT_MODEL", "")
    max_output_tokens = int(os.getenv("EXTRACTION_MAX_OUTPUT_TOKENS") or os.getenv("AGENT_MAX_OUTPUT_TOKENS", "2048"))
    if require_credentials and (not api_key or not base_url or not model):
        raise ValueError("Configure EXTRACTION_* or AGENT_API_KEY, AGENT_BASE_URL, and AGENT_MODEL")
    return DraftConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        max_output_tokens=max_output_tokens,
    )


def _validate_candidates(candidates: list[dict[str, Any]], documents: dict[str, Document]) -> None:
    seen_pairs: set[tuple[str, str]] = set()
    for index, candidate in enumerate(candidates, start=1):
        query = candidate.get("query")
        pmid = candidate.get("pmid")
        if not isinstance(query, str) or not isinstance(pmid, str):
            raise ValueError(f"candidate {index}: query and PMID are required")
        if pmid not in documents:
            raise ValueError(f"candidate {index}: PMID {pmid} is not in the corpus")
        pair = (query, pmid)
        if pair in seen_pairs:
            raise ValueError(f"candidate {index}: duplicate query-PMID pair")
        seen_pairs.add(pair)


def _draft_candidate(
    candidate: dict[str, Any],
    document: Document,
    backend: PromptedExtractionBackend,
) -> dict[str, Any]:
    attempt = run_extraction_attempt(backend, candidate["query"], document)
    annotation = None
    if attempt.extraction is not None:
        annotation = {
            "id": candidate["id"],
            "query": candidate["query"],
            "pmid": document.pmid,
            "annotation_status": "draft",
            "extraction": attempt.extraction.model_dump(mode="json"),
        }
    return {
        "candidate_id": candidate["id"],
        "pmid": document.pmid,
        "annotation": annotation,
        "latency_ms": round(attempt.latency_ms, 3),
        "error_kind": attempt.error_kind,
        "error_message": attempt.error_message,
        "raw_output": attempt.raw_output,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content + ("\n" if content else ""))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
