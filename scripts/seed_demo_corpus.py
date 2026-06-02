from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from bioevidence.config import load_settings
from bioevidence.ingestion.normalize import document_to_record
from bioevidence.ingestion.pubmed_client import fetch_pubmed_batch, save_pubmed_artifacts
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.utils.io import save_json, save_jsonl
from bioevidence.utils.text import slugify_text


DEFAULT_TOPICS = (
    "asthma corticosteroids exacerbations randomized trial",
    "type 2 diabetes metformin glycemic control clinical trial",
    "statins cardiovascular risk primary prevention meta analysis",
    "melanoma immunotherapy survival checkpoint inhibitors",
    "dietary sodium hypertension blood pressure randomized trial",
)


@dataclass(frozen=True, slots=True)
class TopicResult:
    query: str
    requested: int
    document_count: int
    pmids: tuple[str, ...]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "requested": self.requested,
            "document_count": self.document_count,
            "pmids": list(self.pmids),
            "artifacts": self.artifacts,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed a small demo PubMed corpus with real abstracts.")
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        help="PubMed query topic. Can be passed multiple times. Defaults to five demo topics.",
    )
    parser.add_argument(
        "--retmax-per-topic",
        type=int,
        default=30,
        help="Number of PubMed records to request for each topic.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/corpora/demo"),
        help="Base directory for raw, processed, and manifest artifacts.",
    )
    parser.add_argument(
        "--combined-stem",
        default="demo",
        help="Stem for the combined processed corpus and manifest.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    topics = tuple(args.topics or DEFAULT_TOPICS)
    if args.retmax_per_topic <= 0:
        raise SystemExit("--retmax-per-topic must be positive")

    settings = load_settings()
    topic_results: list[TopicResult] = []
    combined_documents: dict[str, Document] = {}

    for topic in topics:
        query = Query(text=topic, top_k=args.retmax_per_topic)
        raw_payload, documents = fetch_pubmed_batch(
            query,
            retmax=args.retmax_per_topic,
            settings=settings,
        )
        artifacts = save_pubmed_artifacts(
            query,
            raw_payload,
            documents,
            output_dir=args.output_dir,
            stem=slugify_text(topic),
        )
        for document in documents:
            if document.pmid and document.abstract:
                combined_documents.setdefault(document.pmid, document)
        topic_results.append(
            TopicResult(
                query=topic,
                requested=args.retmax_per_topic,
                document_count=len(documents),
                pmids=tuple(document.pmid for document in documents if document.pmid),
                artifacts={key: str(value) for key, value in artifacts.items()},
            )
        )

    processed_dir = args.output_dir / "processed"
    combined_path = processed_dir / f"{args.combined_stem}.documents.jsonl"
    manifest_path = processed_dir / f"{args.combined_stem}.manifest.json"
    combined = list(combined_documents.values())
    save_jsonl((document_to_record(document) for document in combined), combined_path)
    save_json(
        {
            "source": "pubmed_eutilities",
            "topic_count": len(topics),
            "document_count": len(combined),
            "topics": [result.to_dict() for result in topic_results],
            "combined_documents_jsonl": str(combined_path),
        },
        manifest_path,
    )

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "topic_count": len(topics),
                "document_count": len(combined),
                "combined_documents_jsonl": str(combined_path),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
