from __future__ import annotations

from bioevidence.schemas.document import RetrievedCandidate


def rerank_candidates(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
