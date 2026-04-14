from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bioevidence.schemas.query import Query
from bioevidence.ingestion.pubmed_client import search_pubmed


def main() -> int:
    query = Query(text="sample")
    documents = search_pubmed(query)
    print(f"Fetched {len(documents)} documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
