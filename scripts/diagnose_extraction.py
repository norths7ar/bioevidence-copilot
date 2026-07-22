from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from bioevidence.config import load_settings
from bioevidence.extraction.model_backend import (
    ExtractionBackend,
    create_product_extraction_backend,
    resolve_extraction,
)
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.schemas.document import Document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose one structured evidence extraction attempt.")
    parser.add_argument("--query", required=True, help="Query-focused extraction question.")
    parser.add_argument("--pmid", required=True, help="PMID to diagnose from the local corpus.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/corpora/demo"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluations/extraction_diagnostic.json"),
    )
    return parser


def build_diagnostic_report(query: str, document: Document, backend: ExtractionBackend) -> dict[str, object]:
    resolution = resolve_extraction(backend, query, document)
    return {
        "query": query,
        "pmid": document.pmid,
        "attempted_backend": resolution.attempted_backend,
        "used_backend": resolution.used_backend,
        "fallback_reason": resolution.fallback_reason,
        "failure_details": resolution.failure_details or None,
        "failed_raw_output": resolution.failed_raw_output or None,
        "extraction": resolution.extraction.model_dump(mode="json"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    backend = create_product_extraction_backend(settings)
    if backend is None:
        raise ValueError("Configure EXTRACTION_BACKEND to rules, prompted, or local")
    document = _find_document(load_local_documents(args.data_dir, settings=settings), args.pmid)
    report = build_diagnostic_report(args.query, document, backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Report: {args.output}")
    return 0


def _find_document(documents: Sequence[Document], pmid: str) -> Document:
    try:
        return next(document for document in documents if document.pmid == pmid)
    except StopIteration as exc:
        raise ValueError(f"PMID not found in local corpus: {pmid}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
