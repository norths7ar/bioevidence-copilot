from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bioevidence.evaluation.dataset import EvaluationItem
from bioevidence.evaluation.quality import check_answer_quality
from bioevidence.evaluation.runner import EvaluationItemResult, EvaluationReport
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.evidence import EvidenceRecord
import scripts.run_eval as run_eval_script


def test_run_eval_cli_prints_summary_and_writes_report(tmp_path: Path, capsys, monkeypatch):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        '{"id": "item-1", "query": "asthma corticosteroids", "gold_pmids": ["111"], "reference_answer": "Answer for asthma corticosteroids"}\n',
        encoding="utf-8",
    )
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
        quality_checks=check_answer_quality(
            AnswerBundle(answer_text="Answer for asthma corticosteroids [111]", citations=("111",)),
            (
                EvidenceRecord(
                    pmid="111",
                    title="Title 111",
                    year=2024,
                    journal="Journal",
                    summary="Abstract 111",
                ),
            ),
        ),
        answer_text="Answer for asthma corticosteroids",
        rewritten_query="asthma corticosteroids",
        retrieval_source="local_corpus",
    )
    report = EvaluationReport(
        dataset_path=dataset,
        mode="baseline",
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

    calls = {}

    def fake_run_evaluation(dataset_path, *, mode, data_dir, limit):
        calls["dataset_path"] = dataset_path
        calls["mode"] = mode
        calls["data_dir"] = data_dir
        calls["limit"] = limit
        return report

    monkeypatch.setattr(run_eval_script, "run_evaluation", fake_run_evaluation)

    exit_code = run_eval_script.main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--mode",
            "baseline",
            "--data-dir",
            "data/corpora/demo",
            "--limit",
            "1",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Evaluation report" in captured.out
    assert "hit@k" in captured.out
    assert output.exists()
    assert calls["mode"] == "baseline"
    assert str(calls["data_dir"]) == "data\\corpora\\demo" or str(calls["data_dir"]) == "data/corpora/demo"
    assert calls["limit"] == 1
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["summary"]["items"] == 1
    assert loaded["mode"] == "baseline"
    assert loaded["items"][0]["retrieval_source"] == "local_corpus"
