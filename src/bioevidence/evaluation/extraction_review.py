from __future__ import annotations

import html
import json
from collections.abc import Sequence

from bioevidence.evaluation.extraction_dataset import ExtractionAnnotation


SECTION_TITLES = {
    "asthma": "Asthma and corticosteroids",
    "diabetes": "Type 2 diabetes and metformin",
    "statins": "Statins and primary prevention",
    "melanoma": "Melanoma immunotherapy",
    "sodium": "Dietary sodium and hypertension",
    "negative": "Cross-topic negative controls",
}


def render_extraction_review(annotations: Sequence[ExtractionAnnotation]) -> str:
    grouped: dict[str, list[ExtractionAnnotation]] = {key: [] for key in SECTION_TITLES}
    for annotation in annotations:
        section = "negative" if annotation.id.startswith("negative-") else annotation.id.split("-", maxsplit=1)[0]
        grouped.setdefault(section, []).append(annotation)

    lines = [
        "# Evidence extraction pilot review",
        "",
        "> Generated review artifact. Edit the source JSONL, not this file.",
        "",
        "For each item, confirm the query-relative evidence status first, then inspect the structured fields.",
        "A checked item can be promoted from `draft` to `reviewed` only after any requested edits are applied.",
        "",
        "Review choices:",
        "",
        "- `[ ] Accept as written`",
        "- `[ ] Change evidence status`",
        "- `[ ] Change extracted fields`",
        "- `[ ] Needs discussion`",
        "",
    ]
    item_number = 0
    for section, title in SECTION_TITLES.items():
        section_annotations = grouped.get(section, [])
        if not section_annotations:
            continue
        lines.extend([f"## {title}", ""])
        for annotation in section_annotations:
            item_number += 1
            lines.extend(_render_annotation(annotation, item_number))
    return "\n".join(lines).rstrip() + "\n"


def _render_annotation(annotation: ExtractionAnnotation, item_number: int) -> list[str]:
    extraction = annotation.extraction
    extraction_json = json.dumps(extraction.model_dump(mode="json"), indent=2, ensure_ascii=False)
    lines = [
        f"### {item_number}. {annotation.id}",
        "",
        f"- **Current status:** `{extraction.evidence_status.value}`",
        f"- **Study design:** `{extraction.study_design.value}`",
        f"- **Review focus:** {_review_focus(extraction.evidence_status.value)}",
        "- [ ] Accept as written",
        "- [ ] Change evidence status",
        "- [ ] Change extracted fields",
        "- [ ] Needs discussion",
        "",
        f"**Query:** {annotation.query}",
        "",
        f"**PMID:** {annotation.document.pmid}",
        "",
        f"**Title:** {annotation.document.title}",
        "",
    ]
    if extraction.outcomes:
        lines.extend(["**Proposed supporting span(s):**", ""])
        for outcome in extraction.outcomes:
            lines.append(f"> {html.escape(outcome.evidence_span)}")
            lines.append("")
    else:
        lines.extend(["**Proposed supporting span(s):** none", ""])
    lines.extend(
        [
            "**Draft extraction:**",
            "",
            "```json",
            extraction_json,
            "```",
            "",
            "<details>",
            "<summary>Full PubMed abstract</summary>",
            "",
            html.escape(annotation.document.abstract),
            "",
            "</details>",
            "",
            "**Reviewer notes:**",
            "",
            "---",
            "",
        ]
    )
    return lines


def _review_focus(evidence_status: str) -> str:
    if evidence_status == "direct":
        return "Confirm that the reported result directly answers the query rather than only sharing its topic."
    if evidence_status == "indirect":
        return "Decide whether the adjacent evidence should remain indirect or move to direct/none."
    if evidence_status == "none":
        return "Confirm that no query-relevant evidence is present despite any superficial term overlap."
    return "Decide whether the abstract is genuinely too incomplete or ambiguous to classify."
