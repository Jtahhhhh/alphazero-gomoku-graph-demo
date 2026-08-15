import pytest

from azgomoku.semantic import Entity, EntityType, EpistemicClass, Predicate, Provenance, RelationFact


def test_entity_fact_and_provenance_are_json_ready():
    entity = Entity("cell:s:r0c0", EntityType.Cell, "s", {"row": 0, "col": 0}, "canon:cell")
    provenance = Provenance(
        "prov:derived", "s", "structural", source_file="azgomoku/graph.py", source_function="cell_graph"
    )
    fact = RelationFact(
        "fact:contains",
        "state:s",
        Predicate.CONTAINS,
        entity.entity_id,
        None,
        provenance.provenance_id,
        EpistemicClass.DERIVED,
    )
    assert entity.dict()["entity_type"] == "Cell"
    assert fact.dict()["predicate"] == "CONTAINS"
    assert fact.dict()["epistemic_class"] == "DERIVED"
    assert provenance.dict()["source_function"] == "cell_graph"


def test_relation_fact_requires_object_xor_value():
    common = ("fact:x", "subject", Predicate.HAS_WEIGHT)
    with pytest.raises(ValueError, match="exactly one"):
        RelationFact(*common, None, None, "prov:x", EpistemicClass.LEARNED)
    with pytest.raises(ValueError, match="exactly one"):
        RelationFact(*common, "object", 1.0, "prov:x", EpistemicClass.LEARNED)


def test_v1_entity_and_predicate_vocabularies_exclude_pattern_open_end_and_extends():
    assert "Pattern" not in {item.value for item in EntityType}
    assert "OpenEnd" not in {item.value for item in EntityType}
    assert "EXTENDS" not in {item.value for item in Predicate}
    assert "HAS_OPEN_END" not in {item.value for item in Predicate}
