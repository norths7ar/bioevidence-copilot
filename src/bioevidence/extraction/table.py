from __future__ import annotations

from collections.abc import Sequence
from textwrap import shorten

from bioevidence.schemas.evidence import EvidenceRecord


TABLE_COLUMNS = (
    ("pmid", "PMID"),
    ("title", "Title"),
    ("year", "Year"),
    ("journal", "Journal"),
    ("entities", "Entities"),
    ("summary", "Summary"),
    ("relevance_score", "Relevance"),
)


def evidence_table_rows(records: Sequence[EvidenceRecord]) -> list[dict[str, object]]:
    return [
        {
            "pmid": record.pmid,
            "title": record.title,
            "year": record.year,
            "journal": record.journal,
            "entities": list(record.entities),
            "summary": record.summary,
            "relevance_score": round(record.relevance_score, 4),
        }
        for record in records
    ]


def render_evidence_table(records: Sequence[EvidenceRecord]) -> str:
    rows = evidence_table_rows(records)
    if not rows:
        return "Evidence table: (none)"

    formatted_rows = [
        {
            "pmid": str(row["pmid"]),
            "title": shorten(str(row["title"]), width=40, placeholder="..."),
            "year": "" if row["year"] is None else str(row["year"]),
            "journal": shorten(str(row["journal"]), width=24, placeholder="..."),
            "entities": ", ".join(row["entities"]) if row["entities"] else "",
            "summary": shorten(str(row["summary"]), width=72, placeholder="..."),
            "relevance_score": f"{float(row['relevance_score']):.4f}",
        }
        for row in rows
    ]

    widths: dict[str, int] = {}
    for key, header in TABLE_COLUMNS:
        widths[key] = len(header)
        for row in formatted_rows:
            widths[key] = max(widths[key], len(row[key]))

    def _render_header() -> str:
        cells = [f" {header:<{widths[key]}} " for key, header in TABLE_COLUMNS]
        return "|" + "|".join(cells) + "|"

    def _render_separator() -> str:
        return "+" + "+".join("-" * (widths[key] + 2) for key, _ in TABLE_COLUMNS) + "+"

    def _render_row(values: dict[str, str]) -> str:
        cells = [f" {values[key]:<{widths[key]}} " for key, _ in TABLE_COLUMNS]
        return "|" + "|".join(cells) + "|"

    lines = [_render_header(), _render_separator()]
    lines.extend(_render_row(row) for row in formatted_rows)
    return "\n".join(lines)
