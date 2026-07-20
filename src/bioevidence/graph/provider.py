from __future__ import annotations

from functools import cached_property
from typing import Any, Protocol

from bioevidence.config import Settings
from bioevidence.graph.entity_linking import EntityLinker, load_nodes_from_neo4j, normalize_text
from bioevidence.graph.models import GraphDiscoveryResult, KGPathNode, KGPathRecord


class GraphDiscoveryProvider(Protocol):
    def discover(self, query: str) -> GraphDiscoveryResult: ...

    def close(self) -> None: ...


class DisabledGraphProvider:
    def discover(self, query: str) -> GraphDiscoveryResult:
        return GraphDiscoveryResult(query=query, status="disabled")

    def close(self) -> None:
        return None


class Neo4jGraphProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    @cached_property
    def driver(self) -> Any:
        try:
            from neo4j import GraphDatabase
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install the graph extra to enable Neo4j discovery") from exc
        if not self._settings.graph_password:
            raise RuntimeError("BIOEVIDENCE_GRAPH_PASSWORD is required when graph discovery is enabled")
        driver = GraphDatabase.driver(
            self._settings.graph_uri,
            auth=(self._settings.graph_user, self._settings.graph_password),
        )
        driver.verify_connectivity()
        return driver

    @cached_property
    def entity_linker(self) -> EntityLinker:
        with self.driver.session(database=self._settings.graph_database) as session:
            return EntityLinker(load_nodes_from_neo4j(session))

    def discover(self, query: str) -> GraphDiscoveryResult:
        linked_entities = tuple(self.entity_linker.link(query, top_k=3))
        if not linked_entities:
            return GraphDiscoveryResult(
                query=query,
                status="empty",
                diagnostics={"reason": "no_linked_entities"},
            )
        with self.driver.session(database=self._settings.graph_database) as session:
            result = session.run(
                """
                UNWIND $anchor_ids AS anchor_id
                MATCH (source {id: anchor_id})-[relationship]-(target)
                WHERE source.name IS NOT NULL AND target.name IS NOT NULL
                RETURN
                  source.id AS source_id,
                  source.name AS source_name,
                  labels(source)[0] AS source_label,
                  type(relationship) AS relationship_type,
                  target.id AS target_id,
                  target.name AS target_name,
                  labels(target)[0] AS target_label
                ORDER BY source_id, relationship_type, target_name, target_id
                LIMIT $limit
                """,
                anchor_ids=[entity.id for entity in linked_entities],
                limit=self._settings.graph_max_paths,
            )
            paths = tuple(_one_hop_record(record) for record in result)
        expanded_queries = build_expansion_queries(
            query,
            paths,
            max_queries=self._settings.graph_max_expansion_queries,
        )
        return GraphDiscoveryResult(
            query=query,
            linked_entities=linked_entities,
            paths=paths,
            expanded_queries=expanded_queries,
            status="ready",
            diagnostics={
                "linked_entity_count": len(linked_entities),
                "path_count": len(paths),
                "expansion_query_count": len(expanded_queries),
            },
        )

    def close(self) -> None:
        if "driver" in self.__dict__:
            self.driver.close()


def create_graph_provider(settings: Settings) -> GraphDiscoveryProvider:
    if not settings.graph_enabled:
        return DisabledGraphProvider()
    return Neo4jGraphProvider(settings)


def build_expansion_queries(
    query: str,
    paths: tuple[KGPathRecord, ...],
    *,
    max_queries: int,
) -> tuple[str, ...]:
    normalized_query = normalize_text(query)
    terms: list[str] = []
    for path in paths:
        if not path.nodes:
            continue
        terminal = path.nodes[-1].name.strip()
        if not terminal or normalize_text(terminal) in normalized_query:
            continue
        if terminal.casefold() not in {term.casefold() for term in terms}:
            terms.append(terminal)
        if len(terms) >= max_queries:
            break
    return tuple(f"{query} {term}" for term in terms)


def _one_hop_record(record: Any) -> KGPathRecord:
    source = KGPathNode(id=record["source_id"], name=record["source_name"], label=record["source_label"])
    target = KGPathNode(id=record["target_id"], name=record["target_name"], label=record["target_label"])
    relationship = record["relationship_type"]
    return KGPathRecord(
        nodes=(source, target),
        relationships=(relationship,),
        hop_count=1,
        pattern=f"{source.label} -{relationship}-> {target.label}",
    )
