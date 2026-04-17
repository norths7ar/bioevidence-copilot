from __future__ import annotations

from bioevidence.schemas.document import RetrievedCandidate


def rerank_candidates(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    ranked_candidates = sorted(candidates, key=lambda candidate: (-candidate.score, candidate.document.pmid))
    return [
        RetrievedCandidate(document=candidate.document, score=candidate.score, rank=index + 1)
        for index, candidate in enumerate(ranked_candidates)
    ]
