from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from bioevidence.graph.models import EntityLinkCandidate, KGNode


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def normalize_text(value: str) -> str:
    value = re.sub(r"(?i)'s\b", "", value)
    return " ".join(TOKEN_PATTERN.findall(value.casefold()))


def token_set(value: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(value.casefold()))


def best_window_similarity(query: str, candidate_name: str) -> float:
    return _best_window_similarity(normalize_text(query).split(), normalize_text(candidate_name))


def _best_window_similarity(query_tokens: Sequence[str], normalized_name: str) -> float:
    name_tokens = normalized_name.split()
    if not query_tokens or not name_tokens:
        return 0.0
    window_size = min(len(name_tokens), len(query_tokens))
    windows = (
        " ".join(query_tokens[index : index + window_size])
        for index in range(0, len(query_tokens) - window_size + 1)
    )
    normalized_name = " ".join(name_tokens)
    return max(SequenceMatcher(None, window, normalized_name).ratio() for window in windows)


@dataclass(frozen=True, slots=True)
class _NormalizedNode:
    node: KGNode
    name: str
    tokens: frozenset[str]


class EntityLinker:
    def __init__(self, nodes: Iterable[KGNode], *, min_score: float = 0.72) -> None:
        self._nodes = tuple(
            _NormalizedNode(
                node=node,
                name=normalized_name,
                tokens=frozenset(normalized_name.split()),
            )
            for node in nodes
            if (normalized_name := normalize_text(node.name))
        )
        self._min_score = min_score

    def link(
        self,
        text: str,
        *,
        top_k: int = 10,
        labels: Sequence[str] | None = None,
    ) -> list[EntityLinkCandidate]:
        if top_k <= 0:
            return []
        normalized_text = normalize_text(text)
        if not normalized_text:
            return []
        query_token_sequence = tuple(normalized_text.split())
        query_tokens = frozenset(query_token_sequence)
        allowed_labels = set(labels) if labels else None
        candidates = [
            candidate
            for node in self._nodes
            if allowed_labels is None or node.node.label in allowed_labels
            if (candidate := self._score_node(normalized_text, query_token_sequence, query_tokens, node)) is not None
        ]
        candidates.sort(key=lambda item: (-item.score, item.label, item.name.casefold(), item.id))
        return candidates[:top_k]

    def _score_node(
        self,
        normalized_text: str,
        query_token_sequence: tuple[str, ...],
        query_tokens: frozenset[str],
        normalized_node: _NormalizedNode,
    ) -> EntityLinkCandidate | None:
        node = normalized_node.node
        normalized_name = normalized_node.name
        if normalized_text == normalized_name:
            return EntityLinkCandidate(node.id, node.name, node.label, 1.0, "exact", node.name)
        if f" {normalized_name} " in f" {normalized_text} ":
            return EntityLinkCandidate(node.id, node.name, node.label, 0.96, "phrase", node.name)
        if normalized_node.tokens <= query_tokens:
            return EntityLinkCandidate(node.id, node.name, node.label, 0.9, "token", node.name)
        fuzzy_score = _best_window_similarity(query_token_sequence, normalized_name)
        if fuzzy_score >= self._min_score:
            return EntityLinkCandidate(
                node.id,
                node.name,
                node.label,
                round(fuzzy_score, 6),
                "fuzzy",
                node.name,
            )
        return None


def load_nodes_from_neo4j(session: Any, *, labels: Sequence[str] | None = None) -> list[KGNode]:
    if labels:
        result = session.run(
            """
            MATCH (n)
            WHERE n.name IS NOT NULL AND any(label IN labels(n) WHERE label IN $labels)
            RETURN n.id AS id, n.name AS name, labels(n)[0] AS label
            ORDER BY label, name, id
            """,
            labels=list(labels),
        )
    else:
        result = session.run(
            """
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN n.id AS id, n.name AS name, labels(n)[0] AS label
            ORDER BY label, name, id
            """
        )
    return [KGNode(id=record["id"], name=record["name"], label=record["label"]) for record in result]
