from __future__ import annotations

import re
from collections.abc import Mapping

from bioevidence.schemas.document import Document


def normalize_pubmed_record(record: Mapping[str, object]) -> Document:
    pmid = str(record.get("pmid", "")).strip()
    title = str(record.get("title", "")).strip()
    abstract = str(record.get("abstract", "")).strip()
    journal = str(record.get("journal", "")).strip()
    year = _normalize_year(record.get("year"))
    authors = _normalize_authors(record.get("authors"))
    return Document(
        pmid=pmid,
        title=title,
        abstract=abstract,
        journal=journal,
        year=year,
        authors=authors,
        source=str(record.get("source", "pubmed")) or "pubmed",
    )


def document_to_record(document: Document) -> dict[str, object]:
    return {
        "pmid": document.pmid,
        "title": document.title,
        "abstract": document.abstract,
        "journal": document.journal,
        "year": document.year,
        "authors": list(document.authors),
        "source": document.source,
    }


def _normalize_year(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        match = re.search(r"(19|20)\d{2}", value)
        if match:
            return int(match.group(0))
    return None


def _normalize_authors(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, tuple):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),)
