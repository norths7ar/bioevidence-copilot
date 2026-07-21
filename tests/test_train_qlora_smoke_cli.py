import json
import subprocess
import sys
from pathlib import Path

from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.evaluation.extraction_sft import write_sft_dataset
from bioevidence.retrieval.corpus import load_local_documents
from training.evidence_extraction.scripts.train_qlora_smoke import _render_training_text


def test_train_qlora_smoke_dry_run_validates_generated_dataset(tmp_path: Path) -> None:
    annotations = load_extraction_annotations(
        Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
        load_local_documents(Path("data/corpora/demo")),
    )
    write_sft_dataset(annotations, tmp_path, source_dataset="pilot_annotations.jsonl")
    completed = subprocess.run(
        [
            sys.executable,
            "training/evidence_extraction/scripts/train_qlora_smoke.py",
            "--dry-run",
            "--train-file",
            str(tmp_path / "train.jsonl"),
            "--dev-file",
            str(tmp_path / "dev.jsonl"),
        ],
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
    assert payload["config"]["assistant_target_starts_with_json"] is True


def test_training_render_keeps_template_scaffolding_out_of_assistant_target() -> None:
    class FakeTokenizer:
        eos_token = "<eos>"

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert [message["role"] for message in messages] == ["system", "user"]
            assert tokenize is False
            assert add_generation_prompt is True
            return "<assistant>\n"

    rendered = _render_training_text(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
            {"role": "assistant", "content": '  {"evidence_status":"none"}  '},
        ],
        FakeTokenizer(),
    )

    assert rendered == '<assistant>\n{"evidence_status":"none"}<eos>'
    assert "<think>" not in rendered


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
