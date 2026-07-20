from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class KGNode:
    id: str
    name: str
    label: str


@dataclass(frozen=True, slots=True)
class EntityLinkCandidate:
    id: str
    name: str
    label: str
    score: float
    match_type: Literal["exact", "phrase", "token", "fuzzy"]
    matched_text: str


@dataclass(frozen=True, slots=True)
class KGPathNode:
    id: str
    name: str
    label: str


@dataclass(frozen=True, slots=True)
class KGPathRecord:
    nodes: tuple[KGPathNode, ...]
    relationships: tuple[str, ...]
    hop_count: int
    pattern: str


@dataclass(frozen=True, slots=True)
class KGTraversalTrace:
    pattern: str
    cypher: str
    anchors: dict[str, tuple[EntityLinkCandidate, ...]]
    relationship_types: dict[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class KGTraversalResult:
    pattern: str
    records: tuple[KGPathRecord, ...]
    trace: KGTraversalTrace


@dataclass(frozen=True, slots=True)
class GraphDiscoveryResult:
    query: str
    linked_entities: tuple[EntityLinkCandidate, ...] = ()
    paths: tuple[KGPathRecord, ...] = ()
    expanded_queries: tuple[str, ...] = ()
    status: Literal["disabled", "empty", "ready", "unavailable"] = "disabled"
    diagnostics: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "query": self.query,
            "status": self.status,
            "linked_entities": [
                {
                    "id": entity.id,
                    "name": entity.name,
                    "label": entity.label,
                    "score": entity.score,
                    "match_type": entity.match_type,
                }
                for entity in self.linked_entities
            ],
            "paths": [
                {
                    "nodes": [
                        {"id": node.id, "name": node.name, "label": node.label}
                        for node in path.nodes
                    ],
                    "relationships": list(path.relationships),
                    "hop_count": path.hop_count,
                    "pattern": path.pattern,
                }
                for path in self.paths
            ],
            "expanded_queries": list(self.expanded_queries),
            "diagnostics": dict(self.diagnostics),
        }
