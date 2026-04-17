from pathlib import Path

import scripts.run_agent as run_agent_script


class _FakeReport:
    def to_dict(self) -> dict[str, object]:
        return {
            "query": "asthma corticosteroids",
            "answer": "Agent synthesis",
            "comparison": {"branch_count": 1},
        }


def test_run_agent_cli_prints_json_and_writes_output(tmp_path: Path, monkeypatch, capsys):
    output_path = tmp_path / "agent-report.json"
    monkeypatch.setattr(run_agent_script, "run_agent_workflow", lambda query, data_dir=None, settings=None: _FakeReport())

    exit_code = run_agent_script.main([
        "--query",
        "asthma corticosteroids",
        "--output",
        str(output_path),
    ])

    captured = capsys.readouterr()

    assert exit_code == 0
    assert '"query": "asthma corticosteroids"' in captured.out
    assert output_path.exists()
    assert '"branch_count": 1' in output_path.read_text(encoding="utf-8")
