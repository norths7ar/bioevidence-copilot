from pathlib import Path

from bioevidence.evaluation.extraction_dataset import load_extraction_annotations
from bioevidence.evaluation.extraction_review import render_extraction_review
from bioevidence.retrieval.corpus import load_local_documents


def test_render_extraction_review_contains_all_pilot_items() -> None:
    documents = load_local_documents(Path("data/corpora/demo"))
    annotations = load_extraction_annotations(
        Path("data/evaluations/evidence_extraction/pilot_annotations.jsonl"),
        documents,
    )

    report = render_extraction_review(annotations)

    assert report.count("### ") == 20
    assert "## Asthma and corticosteroids" in report
    assert "## Cross-topic negative controls" in report
    assert "At W24, FF/UMEC/VI statistically significantly improved" in report
    assert "<summary>Full PubMed abstract</summary>" in report
    assert "[ ] Accept as written" in report
