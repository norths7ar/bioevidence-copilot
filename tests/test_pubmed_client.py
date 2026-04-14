from __future__ import annotations

import json
from pathlib import Path

from bioevidence.ingestion.normalize import document_to_record, normalize_pubmed_record
from bioevidence.ingestion.pubmed_client import fetch_pubmed_batch, save_pubmed_artifacts, search_pubmed
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query


SEARCH_JSON = {
    "header": {"type": "esearch"},
    "esearchresult": {
        "count": "1",
        "retmax": "1",
        "retstart": "0",
        "idlist": ["12345678"],
        "querytranslation": "alpha",
    },
}

EFETCH_XML = """<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">12345678</PMID>
      <Article>
        <Journal>
          <JournalIssue>
            <PubDate>
              <Year>2024</Year>
            </PubDate>
          </JournalIssue>
          <ISOAbbreviation>BioEvid J</ISOAbbreviation>
        </Journal>
        <ArticleTitle>Sample article title</ArticleTitle>
        <Abstract>
          <AbstractText Label="Background">Background text.</AbstractText>
          <AbstractText Label="Results">Results text.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author>
            <LastName>Smith</LastName>
            <ForeName>Jane</ForeName>
          </Author>
          <Author>
            <CollectiveName>BioEvidence Consortium</CollectiveName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def read(self) -> bytes:
        return self._payload


class _FakeOpener:
    def __init__(self) -> None:
        self.requests: list[str] = []

    def __call__(self, request):
        self.requests.append(request.full_url)
        if "esearch.fcgi" in request.full_url:
            return _FakeResponse(json.dumps(SEARCH_JSON).encode("utf-8"))
        if "efetch.fcgi" in request.full_url:
            return _FakeResponse(EFETCH_XML.encode("utf-8"))
        raise AssertionError(f"Unexpected URL: {request.full_url}")


def test_normalize_pubmed_record_normalizes_core_fields():
    document = normalize_pubmed_record(
        {
            "pmid": 42,
            "title": " Sample title ",
            "abstract": " Abstract text ",
            "journal": " Journal Name ",
            "year": "2024 Jan",
            "authors": [" Jane Smith ", "BioEvidence Consortium"],
        }
    )

    assert document.pmid == "42"
    assert document.title == "Sample title"
    assert document.abstract == "Abstract text"
    assert document.journal == "Journal Name"
    assert document.year == 2024
    assert document.authors == ("Jane Smith", "BioEvidence Consortium")


def test_search_pubmed_parses_documents_from_fake_responses():
    opener = _FakeOpener()

    raw_payload, documents = fetch_pubmed_batch(Query(text="alpha"), opener=opener)

    assert raw_payload["pmids"] == ["12345678"]
    assert raw_payload["retmax"] == 10
    assert raw_payload["query"] == "alpha"
    assert raw_payload["esearch"]["esearchresult"]["querytranslation"] == "alpha"
    assert raw_payload["efetch_xml"].strip().startswith("<?xml")

    assert len(documents) == 1
    document = documents[0]
    assert document.pmid == "12345678"
    assert document.title == "Sample article title"
    assert document.abstract == "Background: Background text. Results: Results text."
    assert document.journal == "BioEvid J"
    assert document.year == 2024
    assert document.authors == ("Jane Smith", "BioEvidence Consortium")
    assert any("esearch.fcgi" in url for url in opener.requests)
    assert any("efetch.fcgi" in url for url in opener.requests)

    search_documents = search_pubmed(Query(text="alpha"), opener=_FakeOpener())
    assert search_documents[0].pmid == "12345678"


def test_save_pubmed_artifacts_writes_raw_and_processed_outputs(tmp_path: Path):
    query = Query(text="alpha beta")
    raw_payload = {
        "query": query.text,
        "retmax": 1,
        "tool": "BioEvidence Copilot",
        "email": "",
        "esearch": SEARCH_JSON,
        "efetch_xml": EFETCH_XML,
        "pmids": ["12345678"],
    }
    documents = [
        Document(
            pmid="12345678",
            title="Sample article title",
            abstract="Background: Background text.",
            journal="BioEvid J",
            year=2024,
            authors=("Jane Smith",),
        )
    ]

    paths = save_pubmed_artifacts(query, raw_payload, documents, output_dir=tmp_path)

    assert paths["esearch_json"].exists()
    assert paths["efetch_xml"].exists()
    assert paths["documents_jsonl"].exists()
    assert paths["manifest"].exists()
    assert json.loads(paths["manifest"].read_text(encoding="utf-8"))["document_count"] == 1
    assert document_to_record(documents[0])["pmid"] == "12345678"
