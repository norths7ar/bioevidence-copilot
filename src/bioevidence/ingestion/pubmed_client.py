from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from bioevidence.config import Settings, load_settings
from bioevidence.ingestion.normalize import document_to_record, normalize_pubmed_record
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query
from bioevidence.utils.io import save_json, save_jsonl, save_text
from bioevidence.utils.text import slugify_text

PUBMED_EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(
    query: Query,
    *,
    retmax: int | None = None,
    opener: Callable[[Request], Any] | None = None,
    settings: Settings | None = None,
) -> list[Document]:
    """Search PubMed and return normalized document records."""
    _, documents = fetch_pubmed_batch(
        query,
        retmax=retmax,
        opener=opener,
        settings=settings,
    )
    return documents


def fetch_pubmed_batch(
    query: Query,
    *,
    retmax: int | None = None,
    opener: Callable[[Request], Any] | None = None,
    settings: Settings | None = None,
) -> tuple[dict[str, Any], list[Document]]:
    """Fetch a PubMed search response and the corresponding article records."""
    settings = settings or load_settings()
    opener = opener or urlopen
    limit = retmax if retmax is not None else query.top_k

    esearch_payload = _fetch_esearch_payload(
        query=query.text,
        retmax=limit,
        opener=opener,
        settings=settings,
    )
    pmids = _extract_pmids(esearch_payload)

    efetch_xml = ""
    documents: list[Document] = []
    if pmids:
        efetch_xml = _fetch_efetch_xml(
            pmids=pmids,
            opener=opener,
            settings=settings,
        )
        documents = _parse_pubmed_xml(efetch_xml)

    raw_payload: dict[str, Any] = {
        "query": query.text,
        "retmax": limit,
        "tool": settings.pubmed_tool_name,
        "email": settings.pubmed_email,
        "esearch": esearch_payload,
        "efetch_xml": efetch_xml,
        "pmids": pmids,
    }
    return raw_payload, documents


def save_pubmed_artifacts(
    query: Query,
    raw_payload: dict[str, Any],
    documents: Sequence[Document],
    *,
    output_dir: Path | None = None,
    stem: str | None = None,
) -> dict[str, Path]:
    """Persist raw and processed PubMed artifacts to disk."""
    settings = load_settings()
    base_dir = output_dir or settings.data_dir
    raw_dir = base_dir / "raw"
    processed_dir = base_dir / "processed"
    slug = stem or slugify_text(query.text)

    search_json_path = raw_dir / f"{slug}.esearch.json"
    efetch_xml_path = raw_dir / f"{slug}.efetch.xml"
    documents_jsonl_path = processed_dir / f"{slug}.documents.jsonl"
    manifest_path = processed_dir / f"{slug}.manifest.json"

    save_json(raw_payload["esearch"], search_json_path)
    save_text(raw_payload["efetch_xml"], efetch_xml_path)
    save_jsonl((document_to_record(document) for document in documents), documents_jsonl_path)
    save_json(
        {
            "query": query.text,
            "retmax": raw_payload.get("retmax"),
            "document_count": len(documents),
            "pmids": list(raw_payload.get("pmids", [])),
            "artifacts": {
                "esearch_json": str(search_json_path),
                "efetch_xml": str(efetch_xml_path),
                "documents_jsonl": str(documents_jsonl_path),
            },
        },
        manifest_path,
    )

    return {
        "esearch_json": search_json_path,
        "efetch_xml": efetch_xml_path,
        "documents_jsonl": documents_jsonl_path,
        "manifest": manifest_path,
    }


def _fetch_esearch_payload(
    *,
    query: str,
    retmax: int,
    opener: Callable[[Request], Any],
    settings: Settings,
) -> dict[str, Any]:
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": retmax,
        "term": query,
        "tool": settings.pubmed_tool_name,
    }
    if settings.pubmed_email:
        params["email"] = settings.pubmed_email

    url = _build_url("esearch.fcgi", params)
    return _fetch_json(url, opener)


def _fetch_efetch_xml(
    *,
    pmids: Sequence[str],
    opener: Callable[[Request], Any],
    settings: Settings,
) -> str:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "tool": settings.pubmed_tool_name,
    }
    if settings.pubmed_email:
        params["email"] = settings.pubmed_email

    url = _build_url("efetch.fcgi", params)
    return _fetch_text(url, opener)


def _build_url(endpoint: str, params: dict[str, Any]) -> str:
    return f"{PUBMED_EUTILS_BASE_URL}/{endpoint}?{urlencode(params, doseq=True)}"


def _fetch_json(url: str, opener: Callable[[Request], Any]) -> dict[str, Any]:
    return json.loads(_fetch_text(url, opener))


def _fetch_text(url: str, opener: Callable[[Request], Any]) -> str:
    request = Request(url, headers={"Accept": "application/json, text/xml, application/xml"})
    with opener(request) as response:
        payload = response.read()
    if isinstance(payload, bytes):
        return payload.decode("utf-8")
    return str(payload)


def _extract_pmids(esearch_payload: dict[str, Any]) -> list[str]:
    esearch_result = esearch_payload.get("esearchresult", {})
    if not isinstance(esearch_result, dict):
        return []
    idlist = esearch_result.get("idlist", [])
    if not isinstance(idlist, list):
        return []
    return [str(pmid).strip() for pmid in idlist if str(pmid).strip()]


def _parse_pubmed_xml(xml_text: str) -> list[Document]:
    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    documents: list[Document] = []
    for article in root.findall("PubmedArticle"):
        record = _parse_pubmed_article(article)
        if record is not None:
            documents.append(normalize_pubmed_record(record))
    return documents


def _parse_pubmed_article(article: ET.Element) -> dict[str, Any] | None:
    citation = article.find("MedlineCitation")
    if citation is None:
        return None

    article_node = citation.find("Article")
    if article_node is None:
        return None

    return {
        "pmid": _element_text(citation.find("PMID")),
        "title": _element_text(article_node.find("ArticleTitle")),
        "abstract": _extract_abstract(article_node.find("Abstract")),
        "journal": _extract_journal(article_node),
        "year": _extract_year(article_node),
        "authors": _extract_authors(article_node),
        "source": "pubmed",
    }


def _extract_abstract(abstract_node: ET.Element | None) -> str:
    if abstract_node is None:
        return ""

    parts: list[str] = []
    for abstract_text in abstract_node.findall("AbstractText"):
        text = _element_text(abstract_text)
        if not text:
            continue
        label = abstract_text.attrib.get("Label") or abstract_text.attrib.get("NlmCategory")
        if label:
            parts.append(f"{label}: {text}")
        else:
            parts.append(text)
    return " ".join(parts).strip()


def _extract_journal(article_node: ET.Element) -> str:
    journal_node = article_node.find("Journal")
    if journal_node is None:
        return ""

    iso_abbreviation = _element_text(journal_node.find("ISOAbbreviation"))
    if iso_abbreviation:
        return iso_abbreviation

    title = _element_text(journal_node.find("Title"))
    if title:
        return title

    return ""


def _extract_year(article_node: ET.Element) -> int | None:
    pub_date = article_node.find("Journal/JournalIssue/PubDate")
    if pub_date is None:
        return None

    year_text = _element_text(pub_date.find("Year"))
    if year_text.isdigit():
        return int(year_text)

    medline_date = _element_text(pub_date.find("MedlineDate"))
    for token in medline_date.split():
        if len(token) >= 4 and token[:4].isdigit():
            return int(token[:4])
    return None


def _extract_authors(article_node: ET.Element) -> tuple[str, ...]:
    authors: list[str] = []
    for author in article_node.findall("AuthorList/Author"):
        collective_name = _element_text(author.find("CollectiveName"))
        if collective_name:
            authors.append(collective_name)
            continue

        last_name = _element_text(author.find("LastName"))
        fore_name = _element_text(author.find("ForeName"))
        initials = _element_text(author.find("Initials"))

        if last_name and fore_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name:
            authors.append(last_name)
        elif initials:
            authors.append(initials)

    return tuple(authors)


def _element_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()
