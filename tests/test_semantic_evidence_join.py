from azgomoku.semantic import Entity, EntityType, EpistemicClass
from azgomoku.semantic.evidence_schema import (
    EVIDENCE_GENERATOR_VERSION,
    EvidenceOverlay,
    EvidencePredicate,
    make_evidence_fact,
    make_evidence_provenance,
    validate_evidence_overlay,
)


def attention_overlay(edge_id="edge:s:horizontal:0:1"):
    base = {
        "state:s": {"entity_type": "BoardState"},
        "move:s:a1": {"entity_type": "Move"},
        "edge:s:horizontal:0:1": {"entity_type": "StructuralEdge"},
    }
    observation = Entity(
        "attention:s:checkpoint:horizontal:0:1:final",
        EntityType.AttentionObservation,
        "s",
    )
    source = make_evidence_provenance(
        state_id="s",
        source_kind="attention",
        source_file="azgomoku/explanation/model_evidence.py",
        source_function="collect_model_evidence",
        method="final_layer_relation_attention",
        status="observed",
        evidence_generator_version=EVIDENCE_GENERATOR_VERSION,
        base_kg_manifest_sha256="b" * 64,
        model_type="rgat",
        network_mode="eval",
        checkpoint_path="checkpoint.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_iteration=60,
        training_seed=7,
        board_size=6,
        win_length=4,
        layer="final",
        head="all",
        aggregation_method="mean across attention heads",
        edge_id="horizontal:0:1",
    )
    overlay = EvidenceOverlay(entities=[observation], provenance=[source])
    overlay.add_fact(
        make_evidence_fact(
            subject_id=observation.entity_id,
            predicate=EvidencePredicate.OBSERVES,
            object_id=edge_id,
            provenance_id=source.provenance_id,
            epistemic_class=EpistemicClass.DERIVED,
        )
    )
    overlay.add_fact(
        make_evidence_fact(
            subject_id=observation.entity_id,
            predicate=EvidencePredicate.HAS_ATTENTION_WEIGHT,
            value=0.5,
            provenance_id=source.provenance_id,
            epistemic_class=EpistemicClass.LEARNED,
        )
    )
    return overlay, base


def test_attention_resolves_exactly_to_base_structural_edge():
    overlay, base = attention_overlay()
    report = validate_evidence_overlay(overlay, base)
    assert report.valid
    assert report.external_reference_count == 1


def test_unresolved_attention_edge_fails_closed_without_fuzzy_matching():
    overlay, base = attention_overlay("edge:s:horizontal:1:0")
    report = validate_evidence_overlay(overlay, base)
    assert not report.valid
    assert any("dangling evidence object" in error for error in report.errors)

