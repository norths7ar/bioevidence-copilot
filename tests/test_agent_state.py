from bioevidence.agent.state import AgentState
from bioevidence.agent.stop_criteria import should_stop
from bioevidence.schemas.document import Document, RetrievedCandidate
from bioevidence.schemas.evidence import EvidenceRecord
from bioevidence.schemas.query import Query


def _candidate(pmid: str, score: float, rank: int) -> RetrievedCandidate:
    return RetrievedCandidate(
        document=Document(
            pmid=pmid,
            title=f"Title {pmid}",
            abstract=f"Abstract {pmid}",
            journal="Journal",
            year=2024,
        ),
        score=score,
        rank=rank,
    )


def _evidence(pmid: str, score: float) -> EvidenceRecord:
    return EvidenceRecord(
        pmid=pmid,
        title=f"Title {pmid}",
        year=2024,
        journal="Journal",
        entities=("asthma",),
        summary=f"Summary {pmid}",
        relevance_score=score,
    )


def test_agent_state_merges_candidates_and_evidence():
    state = AgentState(query=Query(text="asthma"))

    assert state.record_branch_query(" asthma review ")
    assert not state.record_branch_query("asthma review")

    state.merge_candidates(
        [
            _candidate("111", 0.6, 2),
            _candidate("111", 0.9, 1),
            _candidate("222", 0.8, 3),
        ]
    )
    state.merge_evidence_records(
        [
            _evidence("111", 0.5),
            _evidence("111", 0.8),
            _evidence("222", 0.7),
        ]
    )

    assert state.unique_pmid_count() == 2
    assert state.all_candidates[0].document.pmid == "111"
    assert state.all_candidates[0].score == 0.9
    assert state.evidence_records[0].pmid == "111"
    assert state.evidence_records[0].relevance_score == 0.8


def test_should_stop_when_enough_unique_evidence():
    state = AgentState(query=Query(text="asthma"))
    state.merge_evidence_records(
        [
            _evidence("111", 0.92),
            _evidence("222", 0.83),
            _evidence("333", 0.71),
        ]
    )

    assert should_stop(state, minimum_unique_pmids=3, minimum_relevance_score=0.6)
    assert state.sufficient is True
    assert state.stop_reason == "sufficient_evidence"


def test_should_stop_when_iterations_exhausted():
    state = AgentState(query=Query(text="asthma"), iterations=3, max_iterations=3)

    assert should_stop(state)
    assert state.sufficient is False
    assert state.stop_reason == "max_iterations"
