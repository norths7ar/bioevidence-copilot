from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bioevidence.graph.cypher import quote_identifier
from bioevidence.graph.entity_linking import EntityLinker
from bioevidence.graph.models import (
    EntityLinkCandidate,
    KGPathNode,
    KGPathRecord,
    KGTraversalResult,
    KGTraversalTrace,
)


NODE_PATTERN = re.compile(r"^(?P<label>[A-Za-z][A-Za-z0-9]*)(?:\((?P<anchor>[^)]+)\))?$")
FORWARD_REL_PATTERN = re.compile(r"^-(?P<alias>[A-Z]+)->$")
REVERSE_REL_PATTERN = re.compile(r"^<-(?P<alias>[A-Z]+)-$")
IMPLICIT_ANCHOR_MIN_FUZZY_SCORE = 0.9
DIRECTED_ALIASES = frozenset({"REGULATES"})
GENERIC_IMPLICIT_ANCHOR_NAMES = frozenset(
    {
        "anatomy",
        "biological process",
        "cellular component",
        "compound",
        "disease",
        "gene",
        "genes",
        "molecular function",
        "pathway",
        "pharmacologic class",
        "side effect",
        "symptom",
    }
)


class PathTemplateError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TemplateNode:
    label: str
    anchor_text: str | None = None


@dataclass(frozen=True, slots=True)
class TemplateStep:
    alias: str
    direction: str


@dataclass(frozen=True, slots=True)
class PathTemplate:
    raw: str
    nodes: tuple[TemplateNode, ...]
    steps: tuple[TemplateStep, ...]

    @property
    def hop_count(self) -> int:
        return len(self.steps)


def parse_path_templates(path_text: str) -> list[PathTemplate]:
    return [parse_path_template(segment.strip()) for segment in path_text.split(" AND ") if segment.strip()]


def parse_path_template(path_text: str) -> PathTemplate:
    tokens = path_text.split()
    if len(tokens) < 3 or len(tokens) % 2 == 0:
        raise PathTemplateError(f"Path template must alternate node and relationship tokens: {path_text!r}")
    nodes: list[TemplateNode] = []
    steps: list[TemplateStep] = []
    for index, token in enumerate(tokens):
        if index % 2 == 0:
            match = NODE_PATTERN.match(token)
            if match is None:
                raise PathTemplateError(f"Invalid node token {token!r} in path template {path_text!r}")
            nodes.append(TemplateNode(label=match.group("label"), anchor_text=match.group("anchor")))
            continue
        forward = FORWARD_REL_PATTERN.match(token)
        reverse = REVERSE_REL_PATTERN.match(token)
        if forward is not None:
            steps.append(TemplateStep(alias=forward.group("alias"), direction="forward"))
        elif reverse is not None:
            steps.append(TemplateStep(alias=reverse.group("alias"), direction="reverse"))
        else:
            raise PathTemplateError(f"Invalid relationship token {token!r} in path template {path_text!r}")
    return PathTemplate(raw=path_text, nodes=tuple(nodes), steps=tuple(steps))


def relationship_alias_map(relationship_types: Sequence[str]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for relationship_type in relationship_types:
        grouped[relationship_type.split("_", maxsplit=1)[0]].append(relationship_type)
    return {alias: tuple(sorted(values)) for alias, values in grouped.items()}


def load_relationship_alias_map(session: Any) -> dict[str, tuple[str, ...]]:
    result = session.run("CALL db.relationshipTypes()")
    return relationship_alias_map([record["relationshipType"] for record in result])


class KGPathRetriever:
    def __init__(
        self,
        session: Any,
        entity_linker: EntityLinker,
        relationship_types_by_alias: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._session = session
        self._entity_linker = entity_linker
        self._relationship_types_by_alias = (
            {key: tuple(value) for key, value in relationship_types_by_alias.items()}
            if relationship_types_by_alias is not None
            else load_relationship_alias_map(session)
        )

    def retrieve(
        self,
        *,
        question: str,
        path_text: str,
        top_k_anchors: int = 3,
        limit_per_template: int = 25,
    ) -> list[KGTraversalResult]:
        results: list[KGTraversalResult] = []
        for template in parse_path_templates(path_text):
            cypher, parameters, anchors, relationship_types = build_template_query(
                template,
                question=question,
                entity_linker=self._entity_linker,
                relationship_types_by_alias=self._relationship_types_by_alias,
                top_k_anchors=top_k_anchors,
                limit=limit_per_template,
            )
            records = (
                tuple(
                    path_record_from_neo4j(record, template.raw)
                    for record in self._session.run(cypher, parameters)
                )
                if any(anchors.values())
                else ()
            )
            results.append(
                KGTraversalResult(
                    pattern=template.raw,
                    records=records,
                    trace=KGTraversalTrace(
                        pattern=template.raw,
                        cypher=cypher,
                        anchors=anchors,
                        relationship_types=relationship_types,
                    ),
                )
            )
        return results


def build_template_query(
    template: PathTemplate,
    *,
    question: str,
    entity_linker: EntityLinker,
    relationship_types_by_alias: Mapping[str, Sequence[str]],
    top_k_anchors: int,
    limit: int,
) -> tuple[str, dict[str, Any], dict[str, tuple[EntityLinkCandidate, ...]], dict[str, tuple[str, ...]]]:
    relationship_types = resolve_relationship_types(template, relationship_types_by_alias)
    anchors = resolve_anchors(template, question=question, entity_linker=entity_linker, top_k=top_k_anchors)
    pattern = build_cypher_pattern(template, relationship_types)
    where_clauses: list[str] = []
    parameters: dict[str, Any] = {"limit": limit}
    for key, candidates in anchors.items():
        if not candidates:
            continue
        parameter_name = f"anchor_ids_{key}"
        parameters[parameter_name] = [candidate.id for candidate in candidates]
        where_clauses.append(f"n{key}.id IN ${parameter_name}")
    where_clause = "\nWHERE " + " AND ".join(where_clauses) if where_clauses else ""
    cypher = f"""
    MATCH path = {pattern}{where_clause}
    RETURN
      [node IN nodes(path) | {{id: node.id, name: node.name, label: labels(node)[0]}}] AS nodes,
      [relationship IN relationships(path) | type(relationship)] AS relationships
    LIMIT $limit
    """
    return cypher, parameters, anchors, relationship_types


def resolve_relationship_types(
    template: PathTemplate,
    relationship_types_by_alias: Mapping[str, Sequence[str]],
) -> dict[str, tuple[str, ...]]:
    resolved: dict[str, tuple[str, ...]] = {}
    for step in template.steps:
        values = tuple(relationship_types_by_alias.get(step.alias, ()))
        if not values:
            raise PathTemplateError(f"No Neo4j relationship types found for alias {step.alias!r}")
        resolved[step.alias] = values
    return resolved


def resolve_anchors(
    template: PathTemplate,
    *,
    question: str,
    entity_linker: EntityLinker,
    top_k: int,
) -> dict[str, tuple[EntityLinkCandidate, ...]]:
    anchors: dict[str, tuple[EntityLinkCandidate, ...]] = {}
    first_position_by_label: dict[str, int] = {}
    for index, node in enumerate(template.nodes):
        first_position_by_label.setdefault(node.label, index)
        if node.anchor_text:
            anchors[str(index)] = tuple(entity_linker.link(node.anchor_text, labels=[node.label], top_k=top_k))
    for label, index in first_position_by_label.items():
        if str(index) in anchors:
            continue
        candidates = tuple(
            candidate
            for candidate in entity_linker.link(question, labels=[label], top_k=top_k)
            if is_confident_implicit_anchor(candidate)
        )
        if candidates:
            anchors[str(index)] = candidates
    return anchors


def is_confident_implicit_anchor(candidate: EntityLinkCandidate) -> bool:
    if candidate.matched_text.casefold() in GENERIC_IMPLICIT_ANCHOR_NAMES:
        return False
    return candidate.match_type in {"exact", "phrase", "token"} or candidate.score >= IMPLICIT_ANCHOR_MIN_FUZZY_SCORE


def build_cypher_pattern(template: PathTemplate, relationship_types: Mapping[str, Sequence[str]]) -> str:
    parts = [node_pattern(0, template.nodes[0].label)]
    for index, step in enumerate(template.steps):
        rel_pattern = relationship_pattern(index, relationship_types[step.alias])
        next_node = node_pattern(index + 1, template.nodes[index + 1].label)
        if step.alias not in DIRECTED_ALIASES:
            parts.append(f"-{rel_pattern}-")
        elif step.direction == "forward":
            parts.append(f"-{rel_pattern}->")
        else:
            parts.append(f"<-{rel_pattern}-")
        parts.append(next_node)
    return "".join(parts)


def node_pattern(index: int, label: str) -> str:
    return f"(n{index}:{quote_identifier(label)})"


def relationship_pattern(index: int, relationship_types: Sequence[str]) -> str:
    types = "|".join(quote_identifier(value) for value in relationship_types)
    return f"[r{index}:{types}]"


def path_record_from_neo4j(record: Mapping[str, Any], pattern: str) -> KGPathRecord:
    nodes = tuple(KGPathNode(id=node["id"], name=node["name"], label=node["label"]) for node in record["nodes"])
    relationships = tuple(record["relationships"])
    return KGPathRecord(nodes=nodes, relationships=relationships, hop_count=len(relationships), pattern=pattern)
