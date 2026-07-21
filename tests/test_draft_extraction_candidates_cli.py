import json
import subprocess
import sys


def test_draft_extraction_candidates_dry_run_validates_queue_without_api_calls() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/draft_extraction_candidates.py", "--dry-run", "--limit", "3"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["candidates"] == 3
    assert payload["unique_pmids"] == 3
    assert payload["max_workers"] == 4
    assert isinstance(payload["api_key_configured"], bool)
