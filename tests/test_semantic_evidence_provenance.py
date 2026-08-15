from dataclasses import replace

from azgomoku.semantic.evidence_schema import validate_evidence_overlay
from tests.test_semantic_evidence_join import attention_overlay


def test_attention_provenance_requires_checkpoint_and_edge_metadata():
    overlay, base = attention_overlay()
    provenance_id, source = next(iter(overlay.provenance.items()))
    overlay.provenance[provenance_id] = replace(source, checkpoint_sha256="short", edge_id=None)
    report = validate_evidence_overlay(overlay, base)
    assert not report.valid
    assert any("invalid checkpoint sha256" in error for error in report.errors)
    assert any("attention provenance lacks" in error for error in report.errors)


def test_tactical_truth_predicate_is_rejected_even_with_learned_provenance():
    overlay, base = attention_overlay()
    fact_id, fact = next(iter(overlay.facts.items()))
    overlay.facts[fact_id] = replace(fact, predicate="CREATES")
    report = validate_evidence_overlay(overlay, base)
    assert not report.valid
    assert any("invalid evidence predicate" in error for error in report.errors)

