import json
from pathlib import Path

import numpy as np

from azgomoku.game import GomokuState
from azgomoku.semantic import EntityType, EpistemicClass, Predicate, validate_artifact
from azgomoku.semantic.extract_evidence import extract_evidence
from azgomoku.semantic.extract_proofs import extract_record_proofs
from azgomoku.semantic.extract_state import extract_state
from azgomoku.semantic.extract_tactics import extract_tactics
from investigation.e3b_graph import structural_edges


BENCHMARK = Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl")


def frozen_records():
    return [json.loads(line) for line in BENCHMARK.read_text(encoding="utf-8").splitlines()]


def record(state_id):
    return next(item for item in frozen_records() if item["state_id"] == state_id)


def test_state_extraction_is_derived_and_referentially_valid():
    state = GomokuState.initial(3, 3)
    artifact = extract_state(state)
    assert validate_artifact(artifact, raise_on_error=True).valid
    types = [item.entity_type.value for item in artifact.entities.values()]
    assert types.count(EntityType.BoardState.value) == 1
    assert types.count(EntityType.Cell.value) == 9
    assert types.count(EntityType.Move.value) == 9
    assert EntityType.StructuralEdge.value in types
    assert {fact.epistemic_class for fact in artifact.facts.values()} == {EpistemicClass.DERIVED}
    assert Predicate.PLAYED_AT in {fact.predicate for fact in artifact.facts.values()}


def test_tactical_adapter_materializes_threat_block_and_forced_response_without_pattern():
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, :2] = -1
    state = GomokuState(board, to_play=1, win_length=3)
    artifact = extract_tactics(state)
    assert validate_artifact(artifact, raise_on_error=True).valid
    types = {item.entity_type.value for item in artifact.entities.values()}
    assert EntityType.WinningThreat.value in types
    assert EntityType.DefenseSet.value in types
    assert EntityType.ForcedResponse.value in types
    assert "Pattern" not in types
    predicates = {fact.predicate for fact in artifact.facts.values()}
    assert Predicate.BLOCKS in predicates
    assert Predicate.FORCES in predicates


def test_tactical_vcf_and_no_proof_frozen_boundaries():
    tactical = extract_record_proofs(record("4dca2566ec2be9b6"))
    assert validate_artifact(tactical, raise_on_error=True).valid
    tactical_proofs = [item for item in tactical.entities.values() if item.entity_type == EntityType.Proof]
    assert len(tactical_proofs) == 2
    assert sum(fact.epistemic_class == EpistemicClass.CERTIFIED for fact in tactical.facts.values()) == 2

    vcf = extract_record_proofs(record("b3a6c7628630359d"))
    assert validate_artifact(vcf, raise_on_error=True).valid
    assert any(item.entity_type == EntityType.ProofNode for item in vcf.entities.values())
    assert any(fact.predicate == Predicate.REQUIRES for fact in vcf.facts.values())

    no_proof = extract_record_proofs(record("74c55e1c7c911cc9"))
    assert validate_artifact(no_proof, raise_on_error=True).valid
    assert not any(item.entity_type == EntityType.Proof for item in no_proof.entities.values())
    assert any(fact.epistemic_class == EpistemicClass.EXACT for fact in no_proof.facts.values())
    assert not any(fact.epistemic_class == EpistemicClass.CERTIFIED for fact in no_proof.facts.values())


def test_attention_and_mcts_remain_learned_while_overlap_is_derived():
    item = record("4dca2566ec2be9b6")
    base = extract_record_proofs(item)
    state_item = item["state"]
    state = GomokuState(
        np.asarray(state_item["board"], dtype=np.int8),
        int(state_item["current_player"]),
        int(state_item["last_move"]),
        int(state_item["win_length"]),
    )
    edges = structural_edges(state.size)
    evidence = {"graph_evidence": {"edges": edges}}
    mcts = {
        "available": True,
        "playouts": 10,
        "candidates": [
            {
                "action": 23,
                "row": 3,
                "col": 5,
                "raw_policy_prior": 0.2,
                "search_prior": 0.2,
                "visits": 10,
                "q": 0.75,
                "pi": 1.0,
                "selected": True,
            }
        ],
    }
    artifact = extract_evidence(
        state,
        model_evidence=evidence,
        mcts_trace=mcts,
        checkpoint="model.pt",
        checkpoint_sha="abc123",
        search_config={"playouts": 10, "seed": 7, "mode": "eval", "checkpoint": "abc123"},
        proofs=item["valid_proofs"],
        artifact=base,
    )
    assert validate_artifact(artifact, raise_on_error=True).valid
    weights = [fact for fact in artifact.facts.values() if fact.predicate == Predicate.HAS_WEIGHT]
    overlaps = [fact for fact in artifact.facts.values() if fact.predicate == Predicate.OVERLAPS]
    mcts_values = [
        fact
        for fact in artifact.facts.values()
        if fact.predicate == Predicate.HAS_ACTION_VALUE
        and artifact.entities[fact.subject_id].entity_type == EntityType.MCTSCandidate
    ]
    assert weights and all(fact.epistemic_class == EpistemicClass.LEARNED for fact in weights)
    assert overlaps and all(fact.epistemic_class == EpistemicClass.DERIVED for fact in overlaps)
    assert mcts_values and all(fact.epistemic_class == EpistemicClass.LEARNED for fact in mcts_values)
    assert not any(
        fact.epistemic_class == EpistemicClass.LEARNED
        and fact.predicate in {Predicate.CREATES, Predicate.BLOCKS, Predicate.SUPPORTS}
        for fact in artifact.facts.values()
    )
