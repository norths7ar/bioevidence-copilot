from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, rank_constant: int = 60) -> list[str]:
    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item_id in enumerate(ranking, start=1):
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (rank_constant + rank)
            best_rank[item_id] = min(best_rank.get(item_id, rank), rank)
    return sorted(scores, key=lambda item_id: (-scores[item_id], best_rank[item_id], item_id))
