import numpy as np

from azgomoku.game import GomokuState
from azgomoku.semantic.identity import (
    attention_observation_identity,
    board_state_identity,
    cell_identity,
    forced_response_identity,
    line_window_identity,
    mcts_candidate_identity,
    move_identity,
    proof_identity,
    proof_node_identity,
    structural_edge_identity,
    transform_proof_node_dict,
    winning_threat_identity,
)
from azgomoku.symmetry import transform_action, transform_flat_proof, transform_relation, transform_state


def state():
    return GomokuState(
        np.asarray([[1, 0, -1], [0, 1, 0], [0, -1, 0]], dtype=np.int8),
        to_play=1,
        last_move=7,
        win_length=3,
    )


def test_d4_transform_changes_raw_scope_but_preserves_semantic_canonical_keys():
    original = state()
    symmetry = 2
    transformed = transform_state(original, symmetry)
    action = 8
    mapped_action = transform_action(action, original.size, symmetry)
    assert board_state_identity(original).entity_id != board_state_identity(transformed).entity_id
    assert board_state_identity(original).canonical_key == board_state_identity(transformed).canonical_key
    assert cell_identity(original, action).canonical_key == cell_identity(transformed, mapped_action).canonical_key
    assert move_identity(original, action).canonical_key == move_identity(transformed, mapped_action).canonical_key


def test_d4_line_threat_response_proof_edge_attention_and_mcts_keys_match():
    original = state()
    symmetry = 5
    transformed = transform_state(original, symmetry)
    relation = "diagonal_down"
    window = [0, 4, 8]
    mapped_relation = transform_relation(relation, symmetry)
    mapped_window = [transform_action(cell, original.size, symmetry) for cell in window]
    completion = 8
    mapped_completion = transform_action(completion, original.size, symmetry)

    line = line_window_identity(original, relation, window)
    mapped_line = line_window_identity(transformed, mapped_relation, mapped_window)
    assert line.canonical_key == mapped_line.canonical_key

    threat = winning_threat_identity(original, 1, relation, window, completion)
    mapped_threat = winning_threat_identity(transformed, 1, mapped_relation, mapped_window, mapped_completion)
    assert threat.canonical_key == mapped_threat.canonical_key
    response = forced_response_identity(original, 5, scope_entity_id=threat.entity_id, scope_canonical_key=threat.canonical_key)
    mapped_response = forced_response_identity(
        transformed,
        transform_action(5, original.size, symmetry),
        scope_entity_id=mapped_threat.entity_id,
        scope_canonical_key=mapped_threat.canonical_key,
    )
    assert response.canonical_key == mapped_response.canonical_key

    proof = {
        "action": 8,
        "concepts": ["immediate_win", "winning_line"],
        "critical_cells": window,
        "critical_relations": [relation],
        "windows": [window],
        "proof_method": "tactical_replay",
        "proof_status": "exact",
    }
    mapped_proof = transform_flat_proof(proof, original.size, symmetry)
    proof_key = proof_identity(original, proof)
    mapped_proof_key = proof_identity(transformed, mapped_proof)
    assert proof_key.canonical_key == mapped_proof_key.canonical_key

    edge = structural_edge_identity(original, relation, 0, 4)
    mapped_edge = structural_edge_identity(
        transformed,
        mapped_relation,
        transform_action(0, original.size, symmetry),
        transform_action(4, original.size, symmetry),
    )
    assert edge.canonical_key == mapped_edge.canonical_key
    attention = attention_observation_identity(original, "sha", relation, 0, 4, "final")
    mapped_attention = attention_observation_identity(
        transformed,
        "sha",
        mapped_relation,
        transform_action(0, original.size, symmetry),
        transform_action(4, original.size, symmetry),
        "final",
    )
    assert attention.canonical_key == mapped_attention.canonical_key
    config = {"playouts": 10, "seed": 7, "mode": "eval", "checkpoint": "sha"}
    assert mcts_candidate_identity(original, config, 8).canonical_key == mcts_candidate_identity(
        transformed, config, mapped_completion
    ).canonical_key

    tree = {
        "player_to_move": 1,
        "move": None,
        "node_type": "OR",
        "terminal": None,
        "children": [
            {"player_to_move": -1, "move": 8, "node_type": "AND", "terminal": "five", "children": []}
        ],
    }
    mapped_tree = transform_proof_node_dict(tree, original.size, symmetry)
    node = proof_node_identity(original, proof_key, (), tree)
    mapped_node = proof_node_identity(transformed, mapped_proof_key, (), mapped_tree)
    assert node.canonical_key == mapped_node.canonical_key
