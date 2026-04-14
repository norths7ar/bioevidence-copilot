from __future__ import annotations

from collections.abc import Mapping

from bioevidence.schemas.document import Document


def normalize_pubmed_record(record: Mapping[str, object]) -> Document:
    pmid = str(record.get("pmid", ""))
    return Document(
        pmid=pmid,
        title=str(record.get("title", "")),
        abstract=str(record.get("abstract", "")),
        journal=str(record.get("journal", "")),
        year=record.get("year") if isinstance(record.get("year"), int) else None,
    )
