import numpy as np

from azgomoku.explanation.rendering import render_knowledge_notice_svg
from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget
from investigation.arena_knowledge import solve_arena_record


def test_arena_solver_record_has_replayed_flat_proof_for_immediate_win():
    board = np.zeros((3, 3), dtype=np.int8)
    board[0, :2] = 1
    board[1, :2] = -1
    state = GomokuState(board, to_play=1, last_move=4, win_length=3)
    record = solve_arena_record(state, GroundTruthBudget(node_cap=10_000, time_cap_ms=1_000))
    assert record["solver"]["status"] == "exact_complete"
    assert record["solver"]["optimal_actions_complete"]
    assert record["valid_proofs"]
    assert all(proof["proof_status"] == "exact" for proof in record["valid_proofs"])


def test_notice_svg_does_not_invent_solver_overlay():
    record = {
        "state_id": "unknown-arena",
        "state": {"board_size": 3, "board": [[0, 0, 0], [0, 1, 0], [0, 0, -1]]},
        "solver": {"status": "unknown", "optimal_actions_complete": False},
        "valid_proofs": [],
    }
    svg = render_knowledge_notice_svg(record, "budget")
    assert "Contrast intentionally omitted" in svg
    assert "UNKNOWN · NO GROUND-TRUTH PROOF" in svg
    assert 'data-layer="solver-proof"' not in svg
