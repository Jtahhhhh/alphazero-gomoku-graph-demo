from pathlib import Path

from investigation.evaluate_h1 import critical_ids
from investigation.semantic_xai import (
    BaseKGIndex,
    _incoming_node_scores,
    aggregate_multi_proof,
    proof_semantic_targets,
)
from investigation.e3b_graph import structural_edges


def test_incoming_node_attention_convention_is_target_only_and_locked():
    edges = [
        {"source": 0, "target": 1},
        {"source": 2, "target": 1},
        {"source": 1, "target": 2},
    ]
    ids, scores = _incoming_node_scores(edges, [0.2, 0.3, 0.5], 2)
    assert ids == ["0000", "0001", "0002", "0003"]
    assert scores == [0.0, 0.5, 0.5, 0.0]


def test_proof_supported_geometry_reproduces_legacy_critical_edge_target():
    base = BaseKGIndex(Path("semantic_kg"))
    state_id = "4dca2566ec2be9b6"
    proof_id = sorted(base.by_state_type[(state_id, "Proof")])[0]
    semantic = proof_semantic_targets(base, state_id, proof_id)
    proof = base.entities[proof_id]["attributes"]
    legacy = structural_edges(6)
    assert semantic["targets"]["ProofSupportedGeometry"] == critical_ids(legacy, proof)
    assert semantic["lineage_fact_ids"]


def test_multi_proof_existential_and_coverage_are_kept_separate():
    rows = [
        {
            "state_id": "s",
            "phase": "late",
            "iteration": 60,
            "semantic_type": "ProofSupportedGeometry",
            "proof_id": "p1",
            "attention_mass": 0.2,
            "excess_over_structural": -0.1,
            "excess_over_random": 0.0,
            "auprc": 0.3,
            "rank_percentile": 0.4,
            "top_k_recall": 0.5,
            "checkpoint_sha256": "a" * 64,
            "base_kg_manifest_sha256": "b" * 64,
            "evidence_manifest_sha256": "c" * 64,
        },
        {
            "state_id": "s",
            "phase": "late",
            "iteration": 60,
            "semantic_type": "ProofSupportedGeometry",
            "proof_id": "p2",
            "attention_mass": 0.6,
            "excess_over_structural": 0.1,
            "excess_over_random": 0.2,
            "auprc": 0.7,
            "rank_percentile": 0.8,
            "top_k_recall": 0.9,
            "checkpoint_sha256": "a" * 64,
            "base_kg_manifest_sha256": "b" * 64,
            "evidence_manifest_sha256": "c" * 64,
        },
    ]
    result = aggregate_multi_proof(rows)[0]
    assert result["existential_max_alignment"] == 0.6
    assert result["coverage_mean_alignment"] == 0.4
    assert result["applicable_proof_count"] == 2
