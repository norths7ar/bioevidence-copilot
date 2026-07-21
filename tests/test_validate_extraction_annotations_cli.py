import runpy
import sys


def test_validate_extraction_annotations_cli(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["validate_extraction_annotations.py"])

    runpy.run_path("scripts/validate_extraction_annotations.py", run_name="__main__")

    output = capsys.readouterr().out
    assert "Validated annotations: 20" in output
    assert "Evidence status: direct=3, indirect=10, none=7" in output
    assert "Annotation status: draft=20" in output
