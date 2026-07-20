from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

from bioevidence.config import Settings, load_settings
from bioevidence.evaluation.dataset import EvaluationItem, load_dataset
from bioevidence.evaluation.metrics import compute_retrieval_metrics
from bioevidence.graph.provider import GraphDiscoveryProvider, create_graph_provider
from bioevidence.retrieval.corpus import load_local_documents
from bioevidence.retrieval.fusion import reciprocal_rank_fusion
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.workflows.baseline import run_rag_pipeline
from bioevidence.workflows.retrieval_stack import run_retrieval_stack


@dataclass(frozen=True, slots=True)
class GraphGainItemResult:
    id: str
    query: str
    gold_pmids: tuple[str, ...]
    graph_status: str
    expansion_queries: tuple[str, ...]
    baseline_pmids: tuple[str, ...]
    augmented_pmids: tuple[str, ...]
    baseline_metrics: dict[str, float]
    augmented_metrics: dict[str, float]
    new_relevant_pmids: tuple[str, ...]

    @property
    def recall_delta(self) -> float:
        return self.augmented_metrics["recall_at_k"] - self.baseline_metrics["recall_at_k"]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "query": self.query,
            "gold_pmids": list(self.gold_pmids),
            "graph_status": self.graph_status,
            "expansion_queries": list(self.expansion_queries),
            "baseline_pmids": list(self.baseline_pmids),
            "augmented_pmids": list(self.augmented_pmids),
            "baseline_metrics": self.baseline_metrics,
            "augmented_metrics": self.augmented_metrics,
            "recall_delta": self.recall_delta,
            "new_relevant_pmids": list(self.new_relevant_pmids),
        }


@dataclass(frozen=True, slots=True)
class GraphGainReport:
    items: tuple[GraphGainItemResult, ...]

    @property
    def summary(self) -> dict[str, float | int]:
        count = len(self.items)
        if count == 0:
            return {
                "evaluated_queries": 0,
                "baseline_recall_at_k": 0.0,
                "augmented_recall_at_k": 0.0,
                "mean_recall_delta": 0.0,
                "queries_improved": 0,
            }
        return {
            "evaluated_queries": count,
            "baseline_recall_at_k": _mean(item.baseline_metrics["recall_at_k"] for item in self.items),
            "augmented_recall_at_k": _mean(item.augmented_metrics["recall_at_k"] for item in self.items),
            "mean_recall_delta": _mean(item.recall_delta for item in self.items),
            "queries_improved": sum(item.recall_delta > 0 for item in self.items),
        }

    def to_dict(self) -> dict[str, object]:
        return {"summary": self.summary, "items": [item.to_dict() for item in self.items]}


def run_graph_gain_evaluation(
    dataset_path: Path,
    *,
    data_dir: Path,
    settings: Settings | None = None,
    graph_provider: GraphDiscoveryProvider | None = None,
    limit: int | None = None,
) -> GraphGainReport:
    settings = settings or load_settings()
    items = load_dataset(dataset_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        items = items[:limit]
    documents = load_local_documents(data_dir, settings=settings)
    provider = graph_provider or create_graph_provider(settings)
    owns_provider = graph_provider is None
    try:
        results = tuple(_evaluate_item(item, documents, data_dir, settings, provider) for item in items)
    finally:
        if owns_provider:
            provider.close()
    return GraphGainReport(items=results)


def compare_retrieval_gain(
    item: EvaluationItem,
    *,
    graph_status: str,
    expansion_queries: tuple[str, ...],
    baseline_pmids: tuple[str, ...],
    augmented_pmids: tuple[str, ...],
) -> GraphGainItemResult:
    baseline_metrics = compute_retrieval_metrics(baseline_pmids[: item.top_k], item.gold_pmids)
    augmented_metrics = compute_retrieval_metrics(augmented_pmids[: item.top_k], item.gold_pmids)
    baseline_set = set(baseline_pmids[: item.top_k])
    augmented_set = set(augmented_pmids[: item.top_k])
    gold_set = set(item.gold_pmids)
    return GraphGainItemResult(
        id=item.id,
        query=item.query,
        gold_pmids=item.gold_pmids,
        graph_status=graph_status,
        expansion_queries=expansion_queries,
        baseline_pmids=baseline_pmids[: item.top_k],
        augmented_pmids=augmented_pmids[: item.top_k],
        baseline_metrics=baseline_metrics,
        augmented_metrics=augmented_metrics,
        new_relevant_pmids=tuple(sorted((augmented_set - baseline_set) & gold_set)),
    )


def _evaluate_item(
    item: EvaluationItem,
    documents: list[Document],
    data_dir: Path,
    settings: Settings,
    provider: GraphDiscoveryProvider,
) -> GraphGainItemResult:
    query = Query(text=item.query, top_k=item.top_k)
    baseline = run_rag_pipeline(query, documents=documents, data_dir=data_dir, settings=settings)
    baseline_pmids = tuple(candidate.document.pmid for candidate in baseline.retrieved_candidates)
    discovery = provider.discover(item.query)
    rankings: list[tuple[str, ...]] = [baseline_pmids]
    for expanded_query in discovery.expanded_queries:
        _, candidates, _, _ = run_retrieval_stack(
            Query(text=expanded_query, rewritten_text=expanded_query, top_k=item.top_k),
            documents=documents,
            data_dir=data_dir,
            settings=settings,
        )
        rankings.append(tuple(candidate.document.pmid for candidate in candidates))
    augmented_pmids = tuple(reciprocal_rank_fusion(rankings))
    return compare_retrieval_gain(
        item,
        graph_status=discovery.status,
        expansion_queries=discovery.expanded_queries,
        baseline_pmids=baseline_pmids,
        augmented_pmids=augmented_pmids,
    )


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0
