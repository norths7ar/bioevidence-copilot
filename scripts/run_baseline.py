from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path
from typing import Sequence
from urllib.error import URLError

from bioevidence.workflows.baseline import run_rag_pipeline
from bioevidence.config import load_settings
from bioevidence.presentation import build_demo_payload, render_demo_output
from bioevidence.ingestion.pubmed_client import PubMedRequestError
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.query import Query


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the BioEvidence baseline RAG workflow.")
    parser.add_argument(
        "--query",
        type=str,
        default="What evidence exists for asthma corticosteroids?",
        help="Biomedical question to run through the baseline workflow.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/corpora/demo"),
        help="Local corpus data directory. The workflow reads processed/*.documents.jsonl under this path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the JSON baseline report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    query = Query(text=args.query)
    settings = load_settings()
    try:
        result = run_rag_pipeline(query, data_dir=args.data_dir, settings=settings)
    except (PubMedRequestError, URLError, OSError) as exc:
        logging.getLogger(__name__).warning("Offline demo mode: %s", exc)
        answer = AnswerBundle(
            answer_text="PubMed fetch is unavailable in the current environment. The scaffold is ready, but live ingestion is disabled.",
            citations=(),
            evidence_records=(),
            rewritten_query=query.text,
        )
        payload = {
            "query": query.text,
            "rewritten_query": answer.rewritten_query,
            "retrieval_source": "offline_fallback",
            "retrieved_papers": [],
            "evidence_table": [],
            "answer": answer.answer_text,
            "citations": list(answer.citations),
            "evidence_count": len(answer.evidence_records),
        }
        print("Evidence table: (none)")
        print()
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    payload = build_demo_payload(result)
    print(render_demo_output(result))
    print()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Report written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
