from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioevidence.config import load_settings
from bioevidence.ingestion.pubmed_client import fetch_pubmed_batch, save_pubmed_artifacts
from bioevidence.schemas.query import Query
from bioevidence.utils.text import slugify_text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch and store a sample PubMed batch.")
    parser.add_argument("query", nargs="?", default="sample biomedical literature query")
    parser.add_argument("--retmax", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where raw and processed artifacts should be written.",
    )
    args = parser.parse_args(argv)

    settings = load_settings()
    query = Query(text=args.query, top_k=args.retmax)
    raw_payload, documents = fetch_pubmed_batch(query, retmax=args.retmax, settings=settings)
    artifacts = save_pubmed_artifacts(
        query,
        raw_payload,
        documents,
        output_dir=args.output_dir or settings.data_dir,
        stem=slugify_text(query.text),
    )

    print(
        json.dumps(
            {
                "query": query.text,
                "document_count": len(documents),
                "artifacts": {key: str(value) for key, value in artifacts.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
