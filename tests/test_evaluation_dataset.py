from __future__ import annotations

from pathlib import Path

import pytest

from bioevidence.evaluation.dataset import EvaluationItem, load_dataset


def test_load_dataset_parses_jsonl_and_ignores_comments(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                "# comment line",
                "",
                '{"id": "item-1", "query": "asthma corticosteroids", "gold_pmids": ["12345678"], "reference_answer": "Corticosteroids reduce asthma exacerbations.", "top_k": 5}',
                '{"id": "item-2", "query": "melanoma immunotherapy", "gold_citations": "23456789"}',
            ]
        ),
        encoding="utf-8",
    )

    items = load_dataset(dataset)

    assert items == [
        EvaluationItem(
            id="item-1",
            query="asthma corticosteroids",
            gold_pmids=("12345678",),
            reference_answer="Corticosteroids reduce asthma exacerbations.",
            top_k=5,
        ),
        EvaluationItem(
            id="item-2",
            query="melanoma immunotherapy",
            gold_pmids=("23456789",),
            reference_answer=None,
            top_k=10,
        ),
    ]


def test_load_dataset_rejects_invalid_rows(tmp_path: Path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text('{"id": "item-1", "query": "asthma corticosteroids"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="gold_pmids"):
        load_dataset(dataset)

