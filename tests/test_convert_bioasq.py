from __future__ import annotations

import json
from pathlib import Path

import scripts.convert_bioasq as convert_bioasq


def test_convert_bioasq_writes_eval_and_snippet_corpus(tmp_path: Path):
    input_path = tmp_path / "bioasq.json"
    input_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "q1",
                        "body": "Does treatment improve outcomes?",
                        "type": "yesno",
                        "documents": ["http://www.ncbi.nlm.nih.gov/pubmed/111"],
                        "ideal_answer": ["Treatment improved outcomes."],
                        "snippets": [
                            {
                                "document": "http://www.ncbi.nlm.nih.gov/pubmed/111",
                                "beginSection": "title",
                                "text": "Treatment trial title",
                            },
                            {
                                "document": "http://www.ncbi.nlm.nih.gov/pubmed/111",
                                "beginSection": "abstract",
                                "text": "Treatment improved outcomes in the trial.",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    corpus_output_dir = tmp_path / "corpora" / "bioasq"
    eval_output_dir = tmp_path / "evaluations" / "bioasq"
    exit_code = convert_bioasq.main(
        [
            "--input",
            str(input_path),
            "--corpus-output-dir",
            str(corpus_output_dir),
            "--eval-output-dir",
            str(eval_output_dir),
        ]
    )

    eval_path = eval_output_dir / "bioasq13b_eval.jsonl"
    documents_path = corpus_output_dir / "processed" / "bioasq13b_snippets.documents.jsonl"
    assert exit_code == 0
    eval_item = json.loads(eval_path.read_text(encoding="utf-8").splitlines()[0])
    document = json.loads(documents_path.read_text(encoding="utf-8").splitlines()[0])
    assert eval_item["query"] == "Does treatment improve outcomes?"
    assert eval_item["gold_pmids"] == ["111"]
    assert eval_item["reference_answer"] == "Treatment improved outcomes."
    assert document["pmid"] == "111"
    assert document["title"] == "Treatment trial title"
    assert "improved outcomes" in document["abstract"]
