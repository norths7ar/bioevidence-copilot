from __future__ import annotations

from bioevidence.evaluation.quality import check_answer_quality
from bioevidence.generation.answerer import generate_answer
from bioevidence.schemas.answer import AnswerBundle
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def _evidence(pmid: str = "111") -> EvidenceRecord:
    return EvidenceRecord(
        pmid=pmid,
        title="Randomized trial of corticosteroids for asthma",
        year=2024,
        journal="Journal",
        entities=("asthma", "corticosteroids"),
        summary="Corticosteroids reduced exacerbations in a randomized trial.",
        relevance_score=0.91,
    )


def test_quality_check_accepts_supported_citations():
    answer = AnswerBundle(answer_text="Corticosteroids reduced exacerbations [111].", citations=("111",))

    result = check_answer_quality(answer, [_evidence("111")])

    assert result.is_faithful is True
    assert result.supported_citations == ("111",)
    assert result.unsupported_citations == tuple()
    assert result.evidence_metadata[0].study_type == "trial"
    assert result.evidence_metadata[0].effect_direction == "benefit"


def test_quality_check_flags_unsupported_citations():
    answer = AnswerBundle(answer_text="Evidence supports treatment [999].", citations=("999",))

    result = check_answer_quality(answer, [_evidence("111")])

    assert result.is_faithful is False
    assert result.unsupported_citations == ("999",)
    assert result.notes


def test_quality_check_flags_inline_citations_missing_from_citation_list():
    answer = AnswerBundle(answer_text="Evidence supports treatment [111].", citations=tuple())

    result = check_answer_quality(answer, [_evidence("111")])

    assert result.is_faithful is False
    assert result.missing_citations == ("111",)


def test_generate_answer_uses_insufficient_evidence_path():
    answer = generate_answer(Query(text="rare disease therapy"), [])

    result = check_answer_quality(answer, [])

    assert "Insufficient evidence" in answer.answer_text
    assert result.insufficient_evidence is True
    assert result.forced_conclusion_without_evidence is False


def test_quality_check_flags_forced_conclusion_without_evidence():
    answer = AnswerBundle(answer_text="This treatment improves outcomes.", citations=tuple())

    result = check_answer_quality(answer, [])

    assert result.is_faithful is False
    assert result.forced_conclusion_without_evidence is True
