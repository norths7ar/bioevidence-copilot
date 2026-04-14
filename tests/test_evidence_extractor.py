from bioevidence.extraction.evidence_extractor import extract_evidence
from bioevidence.schemas.document import Document
from bioevidence.schemas.query import Query


def test_extract_evidence_maps_documents_to_records():
    records = extract_evidence(Query(text="gamma"), [Document(pmid="1")])

    assert len(records) == 1
    assert records[0].pmid == "1"
