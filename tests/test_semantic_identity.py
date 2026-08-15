from copy import deepcopy

import numpy as np

from azgomoku.game import GomokuState
from azgomoku.semantic.identity import (
    board_state_identity,
    cell_identity,
    line_window_identity,
    move_identity,
    proof_identity,
    stable_digest,
    winning_threat_identity,
)


def state():
    return GomokuState(
        np.asarray([[1, 0, -1], [0, 1, 0], [-1, 0, 0]], dtype=np.int8),
        to_play=1,
        last_move=4,
        win_length=3,
    )


def flat_proof():
    return {
        "action": 8,
        "concepts": ["immediate_win", "winning_line"],
        "critical_cells": [0, 4, 8],
        "critical_relations": ["diagonal_down"],
        "windows": [[0, 4, 8]],
        "proof_method": "tactical_replay",
        "proof_status": "exact",
    }


def test_same_object_same_id_and_different_state_different_raw_id():
    original = state()
    assert cell_identity(original, 5) == cell_identity(original, 5)
    assert move_identity(original, 5) == move_identity(original, 5)
    changed = GomokuState(original.board, to_play=-1, last_move=4, win_length=3)
    assert board_state_identity(original).entity_id != board_state_identity(changed).entity_id
    assert cell_identity(original, 5).entity_id != cell_identity(changed, 5).entity_id


def test_line_and_threat_identity_use_semantic_content_not_list_position():
    original = state()
    line = line_window_identity(original, "diagonal_down", [0, 4, 8])
    reversed_line = line_window_identity(original, "diagonal_down", [8, 4, 0])
    assert line == reversed_line
    threat = winning_threat_identity(original, 1, "diagonal_down", [0, 4, 8], 8)
    assert threat.entity_id.startswith("threat:")
    assert threat.entity_id != line.entity_id


def test_flat_proof_id_is_normalized_and_renderer_order_is_irrelevant():
    original = state()
    proof = flat_proof()
    reordered = deepcopy(proof)
    reordered["concepts"].reverse()
    reordered["critical_cells"].reverse()
    reordered["windows"] = list(reversed(reordered["windows"]))
    assert proof_identity(original, proof) == proof_identity(original, reordered)

    proof_list_a = [proof, {**proof, "action": 5, "critical_cells": [3, 4, 5], "windows": [[3, 4, 5]], "critical_relations": ["horizontal"]}]
    proof_list_b = list(reversed(proof_list_a))
    assert {proof_identity(original, item).entity_id for item in proof_list_a} == {
        proof_identity(original, item).entity_id for item in proof_list_b
    }
    assert stable_digest(proof) != "P1"
