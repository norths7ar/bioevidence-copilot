from bioevidence.evaluation.dataset import EvaluationItem
from bioevidence.evaluation.graph_gain import GraphGainReport, compare_retrieval_gain


def test_graph_gain_reports_new_relevant_pmids_and_recall_delta() -> None:
    item = EvaluationItem(
        id="q1",
        query="Alzheimer APOE evidence",
        gold_pmids=("111", "333"),
        top_k=3,
    )

    result = compare_retrieval_gain(
        item,
        graph_status="ready",
        expansion_queries=("Alzheimer APOE evidence TREM2",),
        baseline_pmids=("111", "222", "444"),
        augmented_pmids=("111", "333", "222"),
    )
    report = GraphGainReport(items=(result,))

    assert result.baseline_metrics["recall_at_k"] == 0.5
    assert result.augmented_metrics["recall_at_k"] == 1.0
    assert result.recall_delta == 0.5
    assert result.new_relevant_pmids == ("333",)
    assert report.summary["queries_improved"] == 1
    assert report.summary["mean_recall_delta"] == 0.5
