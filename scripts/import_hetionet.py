from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from neo4j import GraphDatabase

from bioevidence.config import load_settings
from bioevidence.graph.cypher import quote_identifier


DEFAULT_BATCH_SIZE = 5_000
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ImportDataError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Hetionet v1.0 into the configured Neo4j database.")
    parser.add_argument("--hetionet-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    return parser


def ensure_real_data_file(path: Path) -> None:
    if not path.exists():
        raise ImportDataError(f"Hetionet data file not found: {path}")
    with path.open("rb") as file:
        if file.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            raise ImportDataError(f"{path} is a Git LFS pointer; fetch the real data before importing")


@contextmanager
def open_text_maybe_gzip(path: Path) -> Iterator[TextIO]:
    ensure_real_data_file(path)
    with path.open("rb") as raw_file:
        magic = raw_file.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rt", encoding="utf-8", newline="") as text_file:
            yield text_file
    else:
        with path.open("r", encoding="utf-8", newline="") as text_file:
            yield text_file


def read_label_mapping(path: Path) -> dict[str, str]:
    ensure_real_data_file(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        return {row["metanode"]: row["label"] for row in csv.DictReader(file, delimiter="\t")}


def read_relationship_mapping(types_path: Path, metaedges_path: Path) -> dict[str, str]:
    ensure_real_data_file(types_path)
    ensure_real_data_file(metaedges_path)
    with metaedges_path.open("r", encoding="utf-8", newline="") as file:
        abbreviations = {row["metaedge"]: row["abbreviation"] for row in csv.DictReader(file, delimiter="\t")}
    with types_path.open("r", encoding="utf-8", newline="") as file:
        return {abbreviations[row["metaedge"]]: row["rel_type"] for row in csv.DictReader(file, delimiter="\t")}


def read_nodes(path: Path, labels: dict[str, str]) -> Iterator[dict[str, str]]:
    ensure_real_data_file(path)
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file, delimiter="\t"):
            yield {"id": row["id"], "name": row["name"], "kind": row["kind"], "label": labels[row["kind"]]}


def read_edges(
    path: Path,
    labels: dict[str, str],
    relationship_types: dict[str, str],
) -> Iterator[dict[str, str]]:
    with open_text_maybe_gzip(path) as file:
        for row in csv.DictReader(file, delimiter="\t"):
            source_kind = row["source"].split("::", maxsplit=1)[0]
            target_kind = row["target"].split("::", maxsplit=1)[0]
            yield {
                "source": row["source"],
                "target": row["target"],
                "metaedge": row["metaedge"],
                "source_label": labels[source_kind],
                "target_label": labels[target_kind],
                "rel_type": relationship_types[row["metaedge"]],
            }


def create_indexes(session: Any, labels: Iterable[str]) -> None:
    for label in sorted(set(labels)):
        index_name = f"hetionet_{label.lower()}_id"
        session.run(
            f"CREATE INDEX {quote_identifier(index_name)} IF NOT EXISTS FOR (n:{quote_identifier(label)}) ON (n.id)"
        ).consume()
    session.run("CALL db.awaitIndexes()").consume()


def import_nodes(session: Any, path: Path, labels: dict[str, str], batch_size: int) -> Counter[str]:
    counts: Counter[str] = Counter()
    batches: dict[str, list[dict[str, str]]] = defaultdict(list)
    for node in read_nodes(path, labels):
        label = node.pop("label")
        batches[label].append(node)
        counts[label] += 1
        if len(batches[label]) >= batch_size:
            write_node_batch(session, label, batches.pop(label))
    for label, rows in batches.items():
        write_node_batch(session, label, rows)
    return counts


def write_node_batch(session: Any, label: str, rows: list[dict[str, str]]) -> None:
    session.run(
        f"""
        UNWIND $rows AS row
        MERGE (node:{quote_identifier(label)} {{id: row.id}})
        SET node.name = row.name, node.kind = row.kind
        """,
        rows=rows,
    ).consume()


def import_edges(
    session: Any,
    path: Path,
    labels: dict[str, str],
    relationship_types: dict[str, str],
    batch_size: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    batches: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for edge in read_edges(path, labels, relationship_types):
        shape = (edge.pop("source_label"), edge.pop("target_label"), edge.pop("rel_type"))
        batches[shape].append(edge)
        counts[shape[2]] += 1
        if len(batches[shape]) >= batch_size:
            write_edge_batch(session, shape, batches.pop(shape))
    for shape, rows in batches.items():
        write_edge_batch(session, shape, rows)
    return counts


def write_edge_batch(session: Any, shape: tuple[str, str, str], rows: list[dict[str, str]]) -> None:
    source_label, target_label, relationship_type = shape
    session.run(
        f"""
        UNWIND $rows AS row
        MATCH (source:{quote_identifier(source_label)} {{id: row.source}})
        MATCH (target:{quote_identifier(target_label)} {{id: row.target}})
        MERGE (source)-[relationship:{quote_identifier(relationship_type)}]->(target)
        SET relationship.metaedge = row.metaedge
        """,
        rows=rows,
    ).consume()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size <= 0:
        raise ImportDataError("batch-size must be positive")
    settings = load_settings()
    if not settings.graph_password:
        raise ImportDataError("NEO4J_PASSWORD is required")
    root = args.hetionet_root
    nodes_path = root / "hetnet" / "tsv" / "hetionet-v1.0-nodes.tsv"
    edges_path = root / "hetnet" / "tsv" / "hetionet-v1.0-edges.sif.gz"
    labels_path = root / "hetnet" / "neo4j" / "labels.tsv"
    types_path = root / "hetnet" / "neo4j" / "types.tsv"
    metaedges_path = root / "describe" / "edges" / "metaedges.tsv"
    labels = read_label_mapping(labels_path)
    relationship_types = read_relationship_mapping(types_path, metaedges_path)
    with GraphDatabase.driver(
        settings.graph_uri,
        auth=(settings.graph_user, settings.graph_password),
    ) as driver:
        driver.verify_connectivity()
        with driver.session(database=settings.graph_database) as session:
            create_indexes(session, labels.values())
            node_counts = import_nodes(session, nodes_path, labels, args.batch_size)
            edge_counts = import_edges(session, edges_path, labels, relationship_types, args.batch_size)
    print(f"Imported {sum(node_counts.values())} nodes and {sum(edge_counts.values())} relationships")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
