import pytest

from azgomoku.semantic import (
    Entity,
    EntityType,
    EpistemicClass,
    Predicate,
    Provenance,
    RelationFact,
    SemanticArtifact,
    SemanticValidationError,
    validate_artifact,
)
from azgomoku.semantic.validation import validate_collections


def entities():
    return [
        Entity("state:s", EntityType.BoardState, "s"),
        Entity("move:s:a1", EntityType.Move, "s"),
        Entity("cell:s:r0c1", EntityType.Cell, "s"),
        Entity("threat:s:t", EntityType.WinningThreat, "s"),
        Entity("attention:s:e", EntityType.AttentionObservation, "s"),
    ]


def test_valid_derived_exact_certified_and_learned_facts():
    artifact = SemanticArtifact(entities=entities())
    sources = [
        Provenance("prov:d", "s", "structural", source_function="cell_graph"),
        Provenance("prov:e", "s", "solver", method="full_minimax", status="exact_complete"),
        Provenance(
            "prov:c",
            "s",
            "tactical_replay",
            method="tactical_replay",
            status="replay_passed",
            proof_or_certificate_id="proof:s:p",
        ),
        Provenance(
            "prov:l",
            "s",
            "model_evidence",
            source_function="collect_model_evidence",
            model_checkpoint="checkpoint.pt",
        ),
    ]
    for item in sources:
        artifact.add_provenance(item)
    facts = [
        RelationFact("fact:d", "move:s:a1", Predicate.PLAYED_AT, "cell:s:r0c1", None, "prov:d", EpistemicClass.DERIVED),
        RelationFact("fact:e", "move:s:a1", Predicate.OPTIMAL_IN, "state:s", None, "prov:e", EpistemicClass.EXACT),
        RelationFact("fact:c", "move:s:a1", Predicate.BLOCKS, "threat:s:t", None, "prov:c", EpistemicClass.CERTIFIED),
        RelationFact("fact:l", "attention:s:e", Predicate.HAS_WEIGHT, None, 0.25, "prov:l", EpistemicClass.LEARNED),
    ]
    for fact in facts:
        artifact.add_fact(fact)
    assert validate_artifact(artifact).valid


def test_duplicate_and_dangling_ids_are_rejected():
    entity = Entity("state:s", EntityType.BoardState, "s")
    fact = RelationFact("fact:x", "missing", Predicate.CONTAINS, "also-missing", None, "prov:missing", EpistemicClass.DERIVED)
    report = validate_collections([entity, entity], [fact, fact], [])
    assert not report.valid
    assert any("duplicate entity_id" in error for error in report.errors)
    assert any("duplicate fact_id" in error for error in report.errors)
    assert any("dangling subject" in error for error in report.errors)
    assert any("dangling object" in error for error in report.errors)
    assert any("missing provenance" in error for error in report.errors)


def test_exact_and_certified_require_correct_provenance():
    artifact = SemanticArtifact(entities=entities())
    artifact.add_provenance(Provenance("prov:bad-exact", "s", "solver", method="vcf", status="exact_partial"))
    artifact.add_provenance(Provenance("prov:bad-cert", "s", "tactical_replay", method="tactical_replay", status="exact"))
    artifact.add_fact(RelationFact("fact:exact", "move:s:a1", Predicate.OPTIMAL_IN, "state:s", None, "prov:bad-exact", EpistemicClass.EXACT))
    artifact.add_fact(RelationFact("fact:cert", "move:s:a1", Predicate.SUPPORTS, "threat:s:t", None, "prov:bad-cert", EpistemicClass.CERTIFIED))
    report = validate_artifact(artifact)
    assert any("EXACT fact lacks" in error for error in report.errors)
    assert any("CERTIFIED fact lacks" in error for error in report.errors)


def test_learned_evidence_cannot_assert_tactical_truth_and_heuristic_is_reserved():
    artifact = SemanticArtifact(entities=entities())
    artifact.add_provenance(Provenance("prov:l", "s", "model_evidence", model_checkpoint="model.pt"))
    artifact.add_provenance(Provenance("prov:h", "s", "manual", source_function="invented_rule"))
    artifact.add_fact(RelationFact("fact:l", "attention:s:e", Predicate.CREATES, "threat:s:t", None, "prov:l", EpistemicClass.LEARNED))
    artifact.add_fact(RelationFact("fact:h", "move:s:a1", Predicate.BLOCKS, "threat:s:t", None, "prov:h", EpistemicClass.HEURISTIC))
    with pytest.raises(SemanticValidationError) as raised:
        validate_artifact(artifact, raise_on_error=True)
    assert any("LEARNED fact uses tactical truth" in error for error in raised.value.errors)
    assert any("HEURISTIC emission is reserved" in error for error in raised.value.errors)
