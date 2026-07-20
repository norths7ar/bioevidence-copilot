import pytest

from bioevidence.retrieval.fusion import reciprocal_rank_fusion


def test_reciprocal_rank_fusion_rewards_cross_ranking_support() -> None:
    fused = reciprocal_rank_fusion([("A", "B", "C"), ("B", "D", "A")])

    assert fused[:2] == ["B", "A"]
    assert set(fused) == {"A", "B", "C", "D"}


def test_reciprocal_rank_fusion_rejects_invalid_constant() -> None:
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion([("A",)], rank_constant=0)
