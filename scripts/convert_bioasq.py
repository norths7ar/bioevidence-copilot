from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from bioevidence.ingestion.normalize import document_to_record
from bioevidence.schemas.document import Document
from bioevidence.utils.io import save_json, save_jsonl


_PMID_RE = re.compile(r"/pubmed/(\d+)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert BioASQ Task B data into BioEvidence eval and corpus files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tmp/BioASQ-training13b/training13b.json"),
        help="BioASQ JSON file containing a top-level questions array.",
    )
    parser.add_argument(
        "--corpus-output-dir",
        type=Path,
        default=Path("data/corpora/bioasq"),
        help="Output directory for converted BioASQ snippet corpus artifacts.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=Path,
        default=Path("data/evaluations/bioasq"),
        help="Output directory for converted BioASQ evaluation items.",
    )
    parser.add_argument(
        "--stem",
        default="bioasq13b",
        help="Stem for output files.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Optional cap for converted questions.",
    )
    parser.add_argument(
        "--type",
        action="append",
        dest="question_types",
        help="Optional BioASQ question type filter. Can be passed multiple times.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    questions = _load_questions(args.input)
    if args.question_types:
        allowed_types = {value.strip().lower() for value in args.question_types if value.strip()}
        questions = [question for question in questions if str(question.get("type", "")).lower() in allowed_types]
    if args.max_questions is not None:
        if args.max_questions <= 0:
            raise SystemExit("--max-questions must be positive")
        questions = questions[: args.max_questions]

    eval_items = [_to_eval_item(question) for question in questions]
    eval_items = [item for item in eval_items if item is not None]
    documents = _build_documents(questions)

    processed_dir = args.corpus_output_dir / "processed"
    eval_path = args.eval_output_dir / f"{args.stem}_eval.jsonl"
    documents_path = processed_dir / f"{args.stem}_snippets.documents.jsonl"
    manifest_path = args.corpus_output_dir / f"{args.stem}.manifest.json"

    save_jsonl(eval_items, eval_path)
    save_jsonl((document_to_record(document) for document in documents), documents_path)
    save_json(
        {
            "source": "bioasq_task_b",
            "input": str(args.input),
            "question_count": len(questions),
            "eval_item_count": len(eval_items),
            "document_count": len(documents),
            "eval_jsonl": str(eval_path),
            "documents_jsonl": str(documents_path),
        },
        manifest_path,
    )

    print(
        json.dumps(
            {
                "input": str(args.input),
                "question_count": len(questions),
                "eval_item_count": len(eval_items),
                "document_count": len(documents),
                "eval_jsonl": str(eval_path),
                "documents_jsonl": str(documents_path),
                "manifest": str(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("BioASQ input must contain a questions array")
    return [question for question in questions if isinstance(question, dict)]


def _to_eval_item(question: Mapping[str, Any]) -> dict[str, object] | None:
    query = str(question.get("body", "")).strip()
    question_id = str(question.get("id", "")).strip()
    gold_pmids = _gold_pmids(question)
    if not query or not question_id or not gold_pmids:
        return None
    item: dict[str, object] = {
        "id": question_id,
        "query": query,
        "gold_pmids": list(gold_pmids),
        "reference_answer": _reference_answer(question.get("ideal_answer")),
        "top_k": min(max(len(gold_pmids), 5), 20),
        "source": "bioasq_task_b",
        "question_type": str(question.get("type", "")).strip(),
    }
    if item["reference_answer"] is None:
        del item["reference_answer"]
    return item


def _gold_pmids(question: Mapping[str, Any]) -> tuple[str, ...]:
    pmids: list[str] = []
    for document_url in question.get("documents", []) or []:
        pmid = _extract_pmid(document_url)
        if pmid:
            pmids.append(pmid)
    for snippet in question.get("snippets", []) or []:
        if not isinstance(snippet, Mapping):
            continue
        pmid = _extract_pmid(snippet.get("document"))
        if pmid:
            pmids.append(pmid)
    return _unique(pmids)


def _build_documents(questions: Iterable[Mapping[str, Any]]) -> list[Document]:
    titles: dict[str, str] = {}
    snippets_by_pmid: dict[str, list[str]] = defaultdict(list)

    for question in questions:
        for snippet in question.get("snippets", []) or []:
            if not isinstance(snippet, Mapping):
                continue
            pmid = _extract_pmid(snippet.get("document"))
            text = str(snippet.get("text", "")).strip()
            if not pmid or not text:
                continue
            section = str(snippet.get("beginSection", "")).lower()
            if section == "title" and pmid not in titles:
                titles[pmid] = text
            else:
                snippets_by_pmid[pmid].append(text)

    documents: list[Document] = []
    for pmid in sorted(set(titles) | set(snippets_by_pmid)):
        abstract_parts = _unique(snippets_by_pmid.get(pmid, []))
        title = titles.get(pmid, "")
        if not title and abstract_parts:
            title = abstract_parts[0][:120]
        documents.append(
            Document(
                pmid=pmid,
                title=title,
                abstract=" ".join(abstract_parts),
                journal="",
                year=None,
                authors=tuple(),
                source="bioasq_task_b",
            )
        )
    return documents


def _reference_answer(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return " ".join(parts) if parts else None
    return None


def _extract_pmid(value: object) -> str | None:
    text = str(value or "")
    match = _PMID_RE.search(text)
    if match:
        return match.group(1)
    if text.isdigit():
        return text
    return None


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return tuple(output)


if __name__ == "__main__":
    raise SystemExit(main())
