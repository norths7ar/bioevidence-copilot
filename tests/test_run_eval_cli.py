from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bioevidence.evaluation.dataset import EvaluationItem
from bioevidence.evaluation.runner import EvaluationItemResult, EvaluationReport
import scripts.run_eval as run_eval_script


def test_run_eval_cli_prints_summary_and_writes_report(tmp_path: Path, capsys, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id": "item-1", "query": "asthma corticosteroids", "gold_pmids": ["111"], "reference_answer": "Answer for asthma corticosteroids"}\n', encoding="utf-8")
    output = tmp_path / "report.json"

    item = EvaluationItemResult(
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
    )
    report = EvaluationReport(
        dataset_path=dataset,
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
        items=(item,),
    )

    monkeypatch.setattr(run_eval_script, "run_evaluation", lambda dataset_path: report)

    exit_code = run_eval_script.main(["--dataset", str(dataset), "--output", str(output)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Evaluation report" in captured.out
    assert "hit@k" in captured.out
    assert output.exists()
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["summary"]["items"] == 1
    assert loaded["items"][0]["retrieval_source"] == "local_corpus"
