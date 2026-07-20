from bioevidence.graph.models import KGPathNode, KGPathRecord
from bioevidence.graph.provider import build_expansion_queries


def test_graph_paths_become_distinct_literature_queries() -> None:
    paths = (
        KGPathRecord(
            nodes=(
                KGPathNode("Disease::1", "Alzheimer disease", "Disease"),
                KGPathNode("Gene::1", "APOE", "Gene"),
            ),
            relationships=("ASSOCIATES_DaG",),
            hop_count=1,
            pattern="Disease -ASSOCIATES-> Gene",
        ),
        KGPathRecord(
            nodes=(
                KGPathNode("Disease::1", "Alzheimer disease", "Disease"),
                KGPathNode("Gene::2", "TREM2", "Gene"),
            ),
            relationships=("ASSOCIATES_DaG",),
            hop_count=1,
            pattern="Disease -ASSOCIATES-> Gene",
        ),
    )

    queries = build_expansion_queries("Alzheimer disease mechanisms", paths, max_queries=2)

    assert queries == (
        "Alzheimer disease mechanisms APOE",
        "Alzheimer disease mechanisms TREM2",
    )
