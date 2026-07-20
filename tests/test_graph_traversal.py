from bioevidence.graph.entity_linking import EntityLinker
from bioevidence.graph.models import KGNode
from bioevidence.graph.traversal import build_template_query, parse_path_template, relationship_alias_map


def test_parse_multihop_and_reverse_path_template() -> None:
    template = parse_path_template("Disease -DOWNREGULATES-> Gene <-EXPRESSES- Anatomy")

    assert [node.label for node in template.nodes] == ["Disease", "Gene", "Anatomy"]
    assert [step.direction for step in template.steps] == ["forward", "reverse"]
    assert template.hop_count == 2


def test_relationship_aliases_group_hetionet_types() -> None:
    aliases = relationship_alias_map(["ASSOCIATES_DaG", "PARTICIPATES_GpBP", "PARTICIPATES_GpMF"])

    assert aliases["ASSOCIATES"] == ("ASSOCIATES_DaG",)
    assert aliases["PARTICIPATES"] == ("PARTICIPATES_GpBP", "PARTICIPATES_GpMF")


def test_build_template_query_links_question_anchor() -> None:
    linker = EntityLinker(
        [KGNode(id="Disease::DOID:10652", name="Alzheimer's disease", label="Disease")]
    )
    template = parse_path_template("Disease -ASSOCIATES-> Gene -PARTICIPATES-> BiologicalProcess")

    cypher, parameters, anchors, _ = build_template_query(
        template,
        question="genes associated with Alzheimer's disease",
        entity_linker=linker,
        relationship_types_by_alias={
            "ASSOCIATES": ("ASSOCIATES_DaG",),
            "PARTICIPATES": ("PARTICIPATES_GpBP",),
        },
        top_k_anchors=2,
        limit=5,
    )

    assert "(n0:`Disease`)" in cypher
    assert "-[r0:`ASSOCIATES_DaG`]-" in cypher
    assert parameters["anchor_ids_0"] == ["Disease::DOID:10652"]
    assert anchors["0"][0].label == "Disease"
