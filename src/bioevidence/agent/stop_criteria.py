from __future__ import annotations


def should_stop(evidence_count: int, minimum_evidence: int = 3) -> bool:
    return evidence_count >= minimum_evidence
