from dataclasses import replace
import json

import numpy as np

from azgomoku.game import GomokuState
from azgomoku.oracle_agreement import OracleCase, check_oracle_agreement
from azgomoku.solver import solve_actions
from azgomoku.tactics import threat_moves
from azgomoku.vcf import ProofNode, reduce_vcf_proof, replay_vcf_proof, solve_vcf


def make_state(size, stones, k, to_play=1):
    board = np.zeros((size, size), dtype=np.int8)
    for row, col, player in stones:
        board[row, col] = player
    return GomokuState(board, to_play=to_play, win_length=k)


def test_vcf_candidates_exclude_three_for_k4_and_k5():
    for k, size in ((4, 6), (5, 7)):
        row = size // 2
        initial = [(row, col, 1) for col in range(2, k - 1)]
        three_candidate = make_state(size, initial, k)
        move = row * size + (k - 1)
        classified = {item.move: item for item in threat_moves(three_candidate, 1, "vct")}
        assert classified[move].creates_three
        assert move not in {item.move for item in threat_moves(three_candidate, 1, "vcf")}


def test_immediate_five_and_double_four_produce_replayable_proofs():
    immediate = make_state(6, [(2, col, 1) for col in range(3)], 4)
    result = solve_vcf(immediate)
    assert result.status == "exact_partial" and result.value == 1
    assert result.optimal_actions_complete is False
    assert replay_vcf_proof(immediate, result.proof)
    json.dumps(result.dict())

    cross = make_state(
        7,
        [(3, col, 1) for col in (1, 2, 4)]
        + [(row, 3, 1) for row in (1, 2, 4)],
        5,
    )
    result = solve_vcf(cross)
    assert result.status == "exact_partial"
    assert result.proof.children[0].terminal == "unstoppable_double_threat"
    assert replay_vcf_proof(cross, result.proof)


def test_no_obligation_and_tiny_budget_abstain():
    empty = make_state(6, [], 4)
    assert solve_vcf(empty).status == "unknown"
    limited = solve_vcf(empty, node_cap=0, time_cap_ms=10_000)
    assert limited.status == "unknown"
    assert limited.unknown_reason == "node_cap"
    assert limited.value is None and limited.proof is None


def test_counter_four_breaks_the_attacker_tempo():
    # Attacker's (2,2) makes a horizontal four, but defender's forced reply at
    # (2,3) also completes a vertical four with a next-turn win at (0,3)/(5,3).
    s = make_state(
        6,
        [(2, col, 1) for col in (0, 1)]
        + [(row, 3, -1) for row in (1, 3, 4)],
        4,
    )
    result = solve_vcf(s)
    assert result.status == "unknown"


def test_defender_counter_five_precedes_double_threat_shortcut():
    history = [
        35, 3, 34, 1, 17, 6, 25, 32, 2, 28, 29, 4, 7, 26,
        15, 0, 8, 13, 31, 18, 9, 10, 24, 20, 12, 16, 11, 5,
    ]
    s = GomokuState.initial(6, 4)
    for action in history:
        s = s.play(action)
    oracle = solve_actions(s, deadline_ms=2_000, node_budget=1_000_000)
    result = solve_vcf(s)
    assert oracle.status == "exact"
    assert oracle.action_values[14] == -1
    assert result.status in ("unknown", "exact_partial")
    assert result.optimal_actions is None or 14 not in result.optimal_actions
    check_oracle_agreement(
        [OracleCase(s, tuple(history), 71, oracle)],
        solve_vcf,
        proof_validator=replay_vcf_proof,
    )


def test_replay_rejects_missing_and_child_and_reduce_preserves_h1_shape():
    s = GomokuState(
        np.asarray(
            [
                [1, 0, -1, 0, -1, 1],
                [0, 0, 0, 0, 0, -1],
                [0, 0, 0, 0, -1, 0],
                [0, 0, 0, 0, 1, 0],
                [-1, -1, 1, 0, 1, 0],
                [1, -1, 1, 0, 0, 0],
            ],
            dtype=np.int8,
        ),
        to_play=1,
        win_length=4,
    )
    result = solve_vcf(s)
    assert result.status == "exact_partial"
    assert replay_vcf_proof(s, result.proof)
    flat = reduce_vcf_proof(s, result.proof)
    assert set(("action", "concepts", "critical_cells", "critical_relations", "windows")) <= set(flat)
    assert flat["proof_method"] == "vcf" and flat["proof_status"] == "exact"

    first_and = result.proof.children[0]
    assert first_and.node_type == "AND" and first_and.children
    broken_first = replace(first_and, children=first_and.children[:-1])
    broken = replace(result.proof, children=(broken_first,))
    assert not replay_vcf_proof(s, broken)


def test_oracle_agreement_positive_control_and_zero_false_positives():
    states = [
        GomokuState(
            np.asarray(
                [
                    [1, -1, 1, 1, -1, -1],
                    [0, 1, 1, 0, 1, -1],
                    [-1, -1, -1, 0, 0, 1],
                    [-1, 0, -1, 1, 1, -1],
                    [1, 0, -1, -1, 1, -1],
                    [1, -1, 0, 1, 1, 1],
                ],
                dtype=np.int8,
            ),
            to_play=-1,
            win_length=4,
        ),
    ]
    cases = []
    for s in states:
        oracle = solve_actions(s, deadline_ms=5_000, node_budget=1_000_000)
        assert oracle.status == "exact"
        cases.append(OracleCase(s, (), 23, oracle))
    summary = check_oracle_agreement(
        cases,
        solve_vcf,
        proof_validator=replay_vcf_proof,
        threat_node_cap=100_000,
        threat_time_cap_ms=1_000,
    )
    assert summary.checked_states == 1
    assert summary.claimed_states == 1
    assert summary.checked_actions == 1
