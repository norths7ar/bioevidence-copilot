from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from bioevidence.schemas.document import Document


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def document_text(document: Document) -> str:
    parts = [document.title, document.abstract]
    return " ".join(part for part in parts if part).strip()


def document_tokens(document: Document) -> list[str]:
    return tokenize_text(document_text(document))


def document_bigrams(tokens: Sequence[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def bm25_score(
    query_tokens: Sequence[str],
    document_tokens_list: Sequence[Sequence[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    if not query_tokens or not document_tokens_list:
        return [0.0] * len(document_tokens_list)

    corpus_size = len(document_tokens_list)
    average_length = sum(len(tokens) for tokens in document_tokens_list) / corpus_size
    if average_length == 0:
        return [0.0] * len(document_tokens_list)
    document_frequencies: Counter[str] = Counter()
    for tokens in document_tokens_list:
        document_frequencies.update(set(tokens))

    query_term_counts = Counter(query_tokens)
    scores: list[float] = []
    for tokens in document_tokens_list:
        if not tokens:
            scores.append(0.0)
            continue
        term_counts = Counter(tokens)
        document_length = len(tokens)
        score = 0.0
        for term in query_term_counts:
            term_frequency = term_counts.get(term, 0)
            if not term_frequency:
                continue
            document_frequency = document_frequencies.get(term, 0)
            idf = math.log(1.0 + ((corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)))
            denominator = term_frequency + k1 * (1.0 - b + b * (document_length / average_length))
            score += idf * (term_frequency * (k1 + 1.0)) / denominator
        scores.append(score)
    return scores


def overlap_score(query_tokens: Sequence[str], document_tokens_list: Sequence[Sequence[str]]) -> list[float]:
    if not query_tokens or not document_tokens_list:
        return [0.0] * len(document_tokens_list)

    query_unique = set(query_tokens)
    query_bigrams = document_bigrams(query_tokens)
    scores: list[float] = []
    for tokens in document_tokens_list:
        if not tokens:
            scores.append(0.0)
            continue
        document_unique = set(tokens)
        token_overlap = len(query_unique & document_unique) / len(query_unique)
        if query_bigrams:
            phrase_overlap = len(query_bigrams & document_bigrams(tokens)) / len(query_bigrams)
        else:
            phrase_overlap = 0.0
        scores.append((0.7 * token_overlap) + (0.3 * phrase_overlap))
    return scores


def normalize_scores(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    maximum = max(scores)
    minimum = min(scores)
    if maximum == minimum:
        return [1.0 if score > 0 else 0.0 for score in scores]
    scale = maximum - minimum
    return [(score - minimum) / scale for score in scores]
