import runpy
import sys
from pathlib import Path

import pytest

from scripts.validate_extraction_annotations import _load_candidate_keys


def test_validate_extraction_annotations_cli(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_extraction_annotations.py"])

    runpy.run_path("scripts/validate_extraction_annotations.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "Validated annotations: 20" in output
    assert "Evidence status: direct=3, indirect=10, none=7" in output
    assert "Annotation status: draft=20" in output


def test_load_candidate_keys_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    path.write_text(
        '{"id":"one","query":"query","pmid":"1"}\n{"id":"one","query":"query","pmid":"1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate candidate key"):
        _load_candidate_keys(path)
