import bioevidence.graph.entity_linking as entity_linking
from bioevidence.graph.entity_linking import EntityLinker, normalize_text
from bioevidence.graph.models import KGNode


def _linker() -> EntityLinker:
    return EntityLinker(
        [
            KGNode(id="Disease::DOID:10652", name="Alzheimer's disease", label="Disease"),
            KGNode(id="Disease::DOID:14330", name="Parkinson's disease", label="Disease"),
            KGNode(id="Compound::DB00331", name="Metformin", label="Compound"),
            KGNode(id="Gene::672", name="BRCA1", label="Gene"),
        ]
    )


def test_normalize_text_keeps_biomedical_tokens() -> None:
    assert normalize_text("BRCA1 / Alzheimer's disease") == "brca1 alzheimer disease"


def test_entity_linker_normalizes_nodes_once_and_query_once(monkeypatch) -> None:
    calls = 0
    original = entity_linking.normalize_text

    def counting_normalize(value: str) -> str:
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(entity_linking, "normalize_text", counting_normalize)
    linker = EntityLinker(
        [
            KGNode(id="Gene::1", name="APOE", label="Gene"),
            KGNode(id="Gene::2", name="TREM2", label="Gene"),
        ]
    )

    linker.link("APOE evidence")
    linker.link("TREM2 evidence")

    assert calls == 4


def test_entity_linker_prefers_phrase_match() -> None:
    candidates = _linker().link("genes associated with Alzheimer's disease", top_k=3)

    assert candidates[0].id == "Disease::DOID:10652"
    assert candidates[0].match_type == "phrase"


def test_entity_linker_handles_typo_and_label_filter() -> None:
    assert _linker().link("Parkisons disease genes", top_k=1)[0].id == "Disease::DOID:14330"
    assert _linker().link("metformin binds BRCA1", labels=["Gene"], top_k=1)[0].id == "Gene::672"
