import json
import subprocess
import sys
from pathlib import Path


def test_train_qlora_smoke_dry_run_validates_generated_dataset() -> None:
    completed = subprocess.run(
        [sys.executable, "training/evidence_extraction/scripts/train_qlora_smoke.py", "--dry-run"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["dataset"] == {
        "train_rows": 14,
        "dev_rows": 4,
        "train_unique_pmids": 12,
        "dev_unique_pmids": 2,
    }
    assert payload["config"]["max_steps"] == 5
    assert payload["config"]["response_only_loss"] is True


def test_train_qlora_smoke_dry_run_rejects_pmid_leakage(tmp_path: Path) -> None:
    record = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": "{}"},
        ],
        "metadata": {"pmid": "1", "split": "train"},
    }
    train_file = tmp_path / "train.jsonl"
    dev_file = tmp_path / "dev.jsonl"
    train_file.write_text(json.dumps(record) + "\n", encoding="utf-8")
    record["metadata"]["split"] = "dev"
    dev_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "training/evidence_extraction/scripts/train_qlora_smoke.py",
            "--dry-run",
            "--train-file",
            str(train_file),
            "--dev-file",
            str(dev_file),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode != 0
    assert "PMID leakage" in completed.stderr
