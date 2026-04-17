from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bioevidence.agent.workflow import WorkflowResult
from bioevidence.evaluation.dataset import EvaluationItem
from bioevidence.evaluation.runner import EvaluationItemResult, EvaluationReport, run_evaluation, write_report
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def _build_workflow_result(query_text: str, pmids: tuple[str, ...], citations: tuple[str, ...], source: str) -> WorkflowResult:
    documents = tuple(
        Document(
            pmid=pmid,
            title=f"Title {pmid}",
            abstract=f"Abstract {pmid}",
            journal="Journal",
            year=2024,
        )
        for pmid in pmids
    )
    candidates = tuple(
        RetrievedCandidate(document=document, score=1.0 - index * 0.1, rank=index + 1)
        for index, document in enumerate(documents)
    )
    evidence_records = tuple(
        EvidenceRecord(
            pmid=document.pmid,
            title=document.title,
            year=document.year,
            journal=document.journal,
            entities=("asthma",),
            summary=document.abstract,
            relevance_score=1.0 - index * 0.1,
        )
        for index, document in enumerate(documents)
    )
    answer = AnswerBundle(
        answer_text=f"Answer for {query_text}",
        citations=citations,
        evidence_records=evidence_records,
        rewritten_query=query_text,
    )
    return WorkflowResult(
        query=Query(text=query_text),
        documents=documents,
        retrieved_candidates=candidates,
        evidence_records=evidence_records,
        answer=answer,
        source=source,
    )


def test_run_evaluation_produces_summary_and_items(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                '{"id": "item-1", "query": "asthma corticosteroids", "gold_pmids": ["111"], "reference_answer": "Answer for asthma corticosteroids", "top_k": 1}',
                '{"id": "item-2", "query": "melanoma immunotherapy", "gold_pmids": ["222"]}',
            ]
        ),
        encoding="utf-8",
    )

    def fake_pipeline(query: Query, *, data_dir=None, settings=None):
        del data_dir, settings
        if query.text == "asthma corticosteroids":
            return _build_workflow_result(query.text, ("111", "999"), ("111",), "local_corpus")
        return _build_workflow_result(query.text, ("999", "222"), ("222", "999"), "local_corpus")

    report = run_evaluation(dataset, pipeline=fake_pipeline)

    assert report.summary["items"] == 2
    assert report.summary["reference_items"] == 1
    assert report.summary["mean_hit_at_k"] == 1.0
    assert report.items[0].predicted_pmids == ("111",)
    assert report.items[0].retrieval_metrics["mrr"] == 1.0
    assert report.items[0].evidence_table[0]["pmid"] == "111"
    assert report.items[0].answer_metrics["exact_match"] == 1.0
    assert report.items[1].answer_metrics["exact_match"] is None
    assert report.items[1].citation_metrics["precision"] == 0.5


def test_report_round_trips_to_json(tmp_path: Path):
    report = EvaluationReport(
        dataset_path=Path("data/eval/dataset.jsonl"),
        generated_at=datetime.now(timezone.utc),
        summary={
            "items": 1,
            "reference_items": 1,
            "mean_hit_at_k": 1.0,
            "mean_recall_at_k": 1.0,
            "mean_mrr": 1.0,
            "mean_citation_precision": 1.0,
            "mean_citation_recall": 1.0,
            "mean_citation_f1": 1.0,
            "mean_answer_exact_match": 1.0,
            "mean_answer_token_overlap": 1.0,
        },
        items=(
            EvaluationItemResult(
                item=EvaluationItem(
                    id="item-1",
                    query="asthma corticosteroids",
                    gold_pmids=("111",),
                    reference_answer="Answer for asthma corticosteroids",
                    top_k=1,
                ),
                predicted_pmids=("111",),
                predicted_citations=("111",),
                retrieval_metrics={"hit_at_k": 1.0, "recall_at_k": 1.0, "mrr": 1.0},
                citation_metrics={"precision": 1.0, "recall": 1.0, "f1": 1.0},
                answer_metrics={"exact_match": 1.0, "token_overlap": 1.0},
                evidence_table=(
                    {
                        "pmid": "111",
                        "title": "Title 111",
                        "year": 2024,
                        "journal": "Journal",
                        "entities": ["asthma"],
                        "summary": "Abstract 111",
                        "relevance_score": 1.0,
                    },
                ),
                answer_text="Answer for asthma corticosteroids",
                rewritten_query="asthma corticosteroids",
                retrieval_source="local_corpus",
            ),
        ),
    )

    output = tmp_path / "report.json"
    write_report(report, output)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded["summary"]["items"] == 1
    assert loaded["items"][0]["retrieval_source"] == "local_corpus"
