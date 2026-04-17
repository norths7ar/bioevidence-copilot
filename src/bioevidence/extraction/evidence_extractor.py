from __future__ import annotations

import re
from collections.abc import Sequence

from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "can",
    "could",
    "do",
    "does",
    "during",
    "evidence",
    "exists",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "question",
    "sample",
    "should",
    "the",
    "their",
    "there",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "biomedical",
    "literature",
}


def extract_evidence(
    query: Query,
    items: Sequence[Document | RetrievedCandidate],
) -> list[EvidenceRecord]:
    query_terms = _query_terms(query.rewritten_text or query.text)
    records: list[EvidenceRecord] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, RetrievedCandidate):
            document = item.document
            rank = item.rank or index
            score = item.score
        else:
            document = item
            rank = index
            score = 0.0

        records.append(
            EvidenceRecord(
                pmid=document.pmid,
                title=document.title,
                year=document.year,
                journal=document.journal,
                entities=_extract_entities(query_terms, document),
                summary=_summarize_document(document),
                relevance_score=_relevance_score(score, rank),
            )
        )
    return records


def _query_terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_PATTERN.findall(text.lower()):
        if len(token) < 3 or token in _STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return tuple(terms)


def _extract_entities(query_terms: tuple[str, ...], document: Document) -> tuple[str, ...]:
    searchable_text = " ".join([document.title, document.abstract, document.journal]).lower()
    matched_terms = tuple(term for term in query_terms if term in searchable_text)
    if matched_terms:
        return matched_terms
    return query_terms[:3]


def _summarize_document(document: Document) -> str:
    text = " ".join(document.abstract.split()).strip()
    if not text:
        text = " ".join(document.title.split()).strip()
    if not text:
        return ""
    for candidate in re.split(r"(?<=[.!?])\s+", text):
        summary = candidate.strip().rstrip(".")
        if summary:
            return summary[:200]
    return text[:200].rstrip(".")


def _relevance_score(score: float, rank: int) -> float:
    normalized_score = max(0.0, min(1.0, score))
    rank_component = 1.0 / max(1, rank)
    return round(min(1.0, (0.8 * normalized_score) + (0.2 * rank_component)), 4)
