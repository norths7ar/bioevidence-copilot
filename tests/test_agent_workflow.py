from pathlib import Path

import bioevidence.agent.workflow as workflow_module
from bioevidence.agent.workflow import run_rag_pipeline, run_workflow
from bioevidence.schemas.query import Query


def test_run_rag_pipeline_uses_local_corpus(tmp_path: Path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "alpha.documents.jsonl").write_text(
        """
{"pmid": "12345678", "title": "Corticosteroids for asthma control", "abstract": "Corticosteroids reduce asthma exacerbations.", "journal": "Journal A", "year": 2024, "authors": [], "source": "pubmed"}
""".strip()
        + "\n",
        encoding="utf-8",
    )
    def fake_embed_texts(texts, *, client=None, settings=None):
        del client, settings
        vector_map = {
            "asthma corticosteroids": [1.0, 0.0],
        }
        return [vector_map[text] for text in texts]

    def fake_embed_documents(documents, *, client=None, settings=None):
        del client, settings
        vector_map = {
            "Corticosteroids for asthma control Corticosteroids reduce asthma exacerbations.": [1.0, 0.0],
        }
        return [vector_map[f"{document.title} {document.abstract}"] for document in documents]

    monkeypatch.setattr(workflow_module, "search_pubmed", lambda query, settings=None: (_ for _ in ()).throw(AssertionError("search_pubmed should not be used when local corpus is available")))
    monkeypatch.setattr("bioevidence.retrieval.dense.create_embedding_client", lambda settings: object())
    monkeypatch.setattr("bioevidence.retrieval.dense.embed_documents", fake_embed_documents)
    monkeypatch.setattr("bioevidence.retrieval.dense.embed_texts", fake_embed_texts)

    result = run_rag_pipeline(Query(text="asthma corticosteroids"), data_dir=tmp_path)

    assert result.source == "local_corpus"
    assert result.retrieved_candidates
    assert result.answer.answer_text
    assert result.answer.citations == ("12345678",)
    assert result.retrieved_candidates[0].document.pmid == "12345678"


def test_run_workflow_returns_answer_bundle(tmp_path: Path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "alpha.documents.jsonl").write_text(
        """
{"pmid": "12345678", "title": "Corticosteroids for asthma control", "abstract": "Corticosteroids reduce asthma exacerbations.", "journal": "Journal A", "year": 2024, "authors": [], "source": "pubmed"}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    def fake_embed_texts(texts, *, client=None, settings=None):
        del client, settings
        vector_map = {
            "asthma corticosteroids": [1.0, 0.0],
        }
        return [vector_map[text] for text in texts]

    def fake_embed_documents(documents, *, client=None, settings=None):
        del client, settings
        vector_map = {
            "Corticosteroids for asthma control Corticosteroids reduce asthma exacerbations.": [1.0, 0.0],
        }
        return [vector_map[f"{document.title} {document.abstract}"] for document in documents]

    monkeypatch.setattr("bioevidence.retrieval.dense.create_embedding_client", lambda settings: object())
    monkeypatch.setattr("bioevidence.retrieval.dense.embed_documents", fake_embed_documents)
    monkeypatch.setattr("bioevidence.retrieval.dense.embed_texts", fake_embed_texts)

    answer = run_workflow(Query(text="asthma corticosteroids"), data_dir=tmp_path)

    assert answer.answer_text
    assert answer.citations == ("12345678",)
