from bioevidence.graph.entity_linking import EntityLinker
from bioevidence.graph.models import KGNode
from bioevidence.graph.traversal import (
    KGPathRetriever,
    build_template_query,
    parse_path_template,
    relationship_alias_map,
)


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


def test_same_label_path_anchors_origin_and_discovers_terminal_node() -> None:
    linker = EntityLinker([KGNode(id="Gene::TP53", name="TP53", label="Gene")])
    template = parse_path_template("Gene -REGULATES-> Gene")

    cypher, parameters, anchors, _ = build_template_query(
        template,
        question="genes regulated by TP53",
        entity_linker=linker,
        relationship_types_by_alias={"REGULATES": ("REGULATES_GrG",)},
        top_k_anchors=1,
        limit=5,
    )

    assert set(anchors) == {"0"}
    assert parameters == {"limit": 5, "anchor_ids_0": ["Gene::TP53"]}
    assert "n0.id IN $anchor_ids_0" in cypher
    assert "n1.id IN" not in cypher
    assert "LIMIT $limit" in cypher


def test_retriever_does_not_execute_a_template_without_any_anchor() -> None:
    class Session:
        def run(self, *args, **kwargs):
            raise AssertionError("unanchored Cypher must not execute")

    retriever = KGPathRetriever(
        Session(),
        EntityLinker([]),
        relationship_types_by_alias={"REGULATES": ("REGULATES_GrG",)},
    )

    results = retriever.retrieve(question="unknown entity", path_text="Gene -REGULATES-> Gene")

    assert len(results) == 1
    assert results[0].records == ()
    assert results[0].trace.anchors == {}
