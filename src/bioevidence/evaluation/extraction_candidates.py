from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bioevidence.evaluation.extraction_dataset import ExtractionAnnotation
from bioevidence.extraction.model_backend import build_extraction_messages
from bioevidence.retrieval.scoring import bm25_score, document_tokens, tokenize_text
from bioevidence.schemas.document import Document
from bioevidence.schemas.model_evidence import ModelEvidenceExtraction


@dataclass(frozen=True, slots=True)
class CandidateTopic:
    query: str
    pmids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractionCandidate:
    id: str
    query: str
    document: Document
    source_topic: str
    selection_band: str
    bm25_score: float
    bm25_rank: int

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "query": self.query,
            "pmid": self.document.pmid,
            "source_topic": self.source_topic,
            "selection_band": self.selection_band,
            "bm25_score": round(self.bm25_score, 6),
            "bm25_rank": self.bm25_rank,
            "title": self.document.title,
            "year": self.document.year,
        }


def load_candidate_topics(path: Path) -> list[CandidateTopic]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_topics = payload.get("topics") if isinstance(payload, dict) else None
    if not isinstance(raw_topics, list):
        raise ValueError(f"{path}: topics must be a list")
    topics = []
    for index, raw_topic in enumerate(raw_topics, start=1):
        if not isinstance(raw_topic, dict):
            raise ValueError(f"{path}: topic {index} must be an object")
        query = raw_topic.get("query")
        pmids = raw_topic.get("pmids")
        if not isinstance(query, str) or not query.strip() or not isinstance(pmids, list):
            raise ValueError(f"{path}: topic {index} requires query and pmids")
        topics.append(CandidateTopic(query=query.strip(), pmids=tuple(str(pmid) for pmid in pmids)))
    return topics


def select_expansion_candidates(
    topics: Sequence[CandidateTopic],
    documents: Sequence[Document],
    existing_annotations: Sequence[ExtractionAnnotation],
    *,
    high_per_topic: int = 4,
    broad_per_topic: int = 2,
    hard_negative_per_topic: int = 2,
) -> list[ExtractionCandidate]:
    if min(high_per_topic, broad_per_topic, hard_negative_per_topic) < 0:
        raise ValueError("candidate counts must be non-negative")

    documents_by_pmid = {document.pmid: document for document in documents}
    source_topics_by_pmid: dict[str, list[str]] = {}
    for topic in topics:
        for pmid in topic.pmids:
            source_topics_by_pmid.setdefault(pmid, []).append(topic.query)
    existing_pairs = {(annotation.query, annotation.document.pmid) for annotation in existing_annotations}

    candidates: list[ExtractionCandidate] = []
    for topic in topics:
        topic_documents = [documents_by_pmid[pmid] for pmid in topic.pmids if pmid in documents_by_pmid]
        same_topic_ranked = [
            item for item in _rank_documents(topic.query, topic_documents) if (topic.query, item[0].pmid) not in existing_pairs
        ]
        high = same_topic_ranked[:high_per_topic]
        high_pmids = {document.pmid for document, _, _ in high}
        broad_pool = [item for item in same_topic_ranked[high_per_topic:] if item[0].pmid not in high_pmids]
        broad = _evenly_spaced(broad_pool, broad_per_topic)

        selected_pmids = high_pmids | {document.pmid for document, _, _ in broad}
        cross_topic_documents = [
            document
            for document in documents
            if topic.query not in source_topics_by_pmid.get(document.pmid, []) and document.pmid not in selected_pmids
        ]
        hard_negatives = [
            item
            for item in _rank_documents(topic.query, cross_topic_documents)
            if (topic.query, item[0].pmid) not in existing_pairs
        ][:hard_negative_per_topic]

        candidates.extend(
            _make_candidates(topic.query, high, "topic_high", source_topics_by_pmid)
            + _make_candidates(topic.query, broad, "topic_broad", source_topics_by_pmid)
            + _make_candidates(topic.query, hard_negatives, "cross_topic_hard_negative", source_topics_by_pmid)
        )

    candidate_pairs = [(candidate.query, candidate.document.pmid) for candidate in candidates]
    if len(candidate_pairs) != len(set(candidate_pairs)):
        raise RuntimeError("candidate selection produced duplicate query-PMID pairs")
    return candidates


def build_candidate_manifest(
    candidates: Sequence[ExtractionCandidate],
    *,
    source_corpus: Path,
    existing_annotations: Path,
    high_per_topic: int = 4,
    broad_per_topic: int = 2,
    hard_negative_per_topic: int = 2,
) -> dict[str, Any]:
    return {
        "format": "bioevidence_extraction_candidates_v1",
        "source_corpus": source_corpus.as_posix(),
        "source_corpus_sha256": _canonical_text_sha256(source_corpus),
        "existing_annotations": existing_annotations.as_posix(),
        "selection": {
            "high_per_topic": high_per_topic,
            "broad_per_topic": broad_per_topic,
            "hard_negative_per_topic": hard_negative_per_topic,
        },
        "candidate_pairs": len(candidates),
        "unique_pmids": len({candidate.document.pmid for candidate in candidates}),
        "queries": dict(sorted(Counter(candidate.query for candidate in candidates).items())),
        "selection_bands": dict(sorted(Counter(candidate.selection_band for candidate in candidates).items())),
    }


def _canonical_text_sha256(path: Path) -> str:
    """Hash UTF-8 text with LF newlines so provenance is stable across OS checkouts."""

    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def build_annotation_prompt_records(candidates: Sequence[ExtractionCandidate]) -> list[dict[str, Any]]:
    target_schema = ModelEvidenceExtraction.model_json_schema().get("$id")
    return [
        {
            "messages": build_extraction_messages(candidate.query, candidate.document),
            "metadata": {
                **candidate.to_record(),
                "target_schema": target_schema,
                "label_status": "unlabeled",
            },
        }
        for candidate in candidates
    ]


def _rank_documents(query: str, documents: Sequence[Document]) -> list[tuple[Document, float, int]]:
    token_lists = [document_tokens(document) for document in documents]
    scores = bm25_score(tokenize_text(query), token_lists)
    ranked = sorted(zip(documents, scores, strict=True), key=lambda item: (-item[1], item[0].pmid))
    return [(document, score, rank) for rank, (document, score) in enumerate(ranked, start=1)]


def _evenly_spaced(
    ranked: Sequence[tuple[Document, float, int]],
    count: int,
) -> list[tuple[Document, float, int]]:
    if count <= 0 or not ranked:
        return []
    if len(ranked) <= count:
        return list(ranked)
    indices = [round((index + 1) * (len(ranked) + 1) / (count + 1)) - 1 for index in range(count)]
    return [ranked[max(0, min(position, len(ranked) - 1))] for position in indices]


def _make_candidates(
    query: str,
    ranked: Sequence[tuple[Document, float, int]],
    selection_band: str,
    source_topics_by_pmid: dict[str, list[str]],
) -> list[ExtractionCandidate]:
    return [
        ExtractionCandidate(
            id=f"expansion-{_query_slug(query)}-{document.pmid}",
            query=query,
            document=document,
            source_topic=" | ".join(sorted(source_topics_by_pmid.get(document.pmid, ["unknown"]))),
            selection_band=selection_band,
            bm25_score=score,
            bm25_rank=rank,
        )
        for document, score, rank in ranked
    ]


def _query_slug(query: str) -> str:
    tokens = tokenize_text(query)
    return "-".join(tokens[:3])
