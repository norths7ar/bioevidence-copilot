from __future__ import annotations

from bioevidence.schemas.query import Query


def plan_next_steps(query: Query, evidence_count: int = 0) -> list[str]:
    _ = query
    return ["retrieve", "extract", "generate"] if evidence_count == 0 else ["generate"]
