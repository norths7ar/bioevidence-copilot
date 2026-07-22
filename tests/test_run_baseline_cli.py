from pathlib import Path

import scripts.run_baseline as run_baseline_script
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.workflows import WorkflowResult


def test_run_baseline_cli_passes_top_k_to_workflow(monkeypatch, capsys) -> None:
    captured: dict[str, int] = {}

    def fake_workflow(query, *, data_dir: Path, settings):
        captured["top_k"] = query.top_k
        return WorkflowResult(
            query=query,
            documents=(),
            retrieved_candidates=(),
            evidence_records=(),
            answer=AnswerBundle(answer_text="No evidence", citations=(), evidence_records=()),
            source="local_corpus",
        )

    monkeypatch.setattr(run_baseline_script, "run_rag_pipeline", fake_workflow)

    exit_code = run_baseline_script.main(["--query", "asthma", "--top-k", "3"])

    assert exit_code == 0
    assert captured["top_k"] == 3
    assert '"evidence_count": 0' in capsys.readouterr().out
