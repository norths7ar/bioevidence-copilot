from __future__ import annotations

import json
from pathlib import Path

import scripts.seed_demo_corpus as seed_demo_corpus
from bioevidence.schemas.document import Document


def test_seed_demo_corpus_writes_combined_documents(tmp_path: Path, monkeypatch):
    def fake_fetch_pubmed_batch(query, *, retmax, settings):
        del settings
        document = Document(
            pmid=f"{len(query.text)}",
            title=f"Title for {query.text}",
            abstract="A real abstract would be stored here.",
            journal="Journal",
            year=2024,
            source="pubmed",
        )
        return {"esearch": {}, "efetch_xml": "", "pmids": [document.pmid], "retmax": retmax}, [document]

    monkeypatch.setattr(seed_demo_corpus, "fetch_pubmed_batch", fake_fetch_pubmed_batch)
    monkeypatch.setattr(seed_demo_corpus, "load_settings", lambda: object())

    output_dir = tmp_path / "demo"
    exit_code = seed_demo_corpus.main(
        [
            "--topic",
            "asthma corticosteroids",
            "--topic",
            "statins prevention",
            "--retmax-per-topic",
            "2",
            "--output-dir",
            str(output_dir),
        ]
    )

    combined_path = output_dir / "processed" / "demo.documents.jsonl"
    manifest_path = output_dir / "processed" / "demo.manifest.json"
    lines = combined_path.read_text(encoding="utf-8").splitlines()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert len(lines) == 2
    assert manifest["topic_count"] == 2
    assert manifest["document_count"] == 2
