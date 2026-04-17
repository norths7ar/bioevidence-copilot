from __future__ import annotations

from pathlib import Path

from bioevidence.config import Settings, load_settings
from bioevidence.ingestion.normalize import normalize_pubmed_record
from bioevidence.schemas.document import Document
from bioevidence.utils.io import iter_jsonl


def load_local_documents(
    data_dir: Path | None = None,
    *,
    settings: Settings | None = None,
) -> list[Document]:
    settings = settings or load_settings()
    base_dir = Path(data_dir or settings.data_dir)
    processed_dir = base_dir / "processed"
    if not processed_dir.exists():
        return []

    documents: list[Document] = []
    seen_pmids: set[str] = set()
    for path in sorted(processed_dir.glob("*.documents.jsonl")):
        for record in iter_jsonl(path):
            if not isinstance(record, dict):
                continue
            document = normalize_pubmed_record(record)
            if not document.pmid or document.pmid in seen_pmids:
                continue
            seen_pmids.add(document.pmid)
            documents.append(document)
    return documents
