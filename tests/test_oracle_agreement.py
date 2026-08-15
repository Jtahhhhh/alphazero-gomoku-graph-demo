import numpy as np
import pytest

from azgomoku.game import GomokuState
from azgomoku.oracle_agreement import (
    OracleCase,
    ThreatClaim,
    check_oracle_agreement,
    generate_oracle_cases,
    threat_solver_stub,
)
from azgomoku.solver import solve_actions


def tactical_case():
    board = np.asarray(
        [
            [1, 1, 1, 0],
            [-1, -1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int8,
    )
    state = GomokuState(board, to_play=1, win_length=4)
    # Keep this correctness fixture insensitive to transient full-suite CPU load;
    # the test is about harness semantics, not a 2-second performance boundary.
    oracle = solve_actions(state, deadline_ms=10_000, node_budget=1_000_000)
    assert oracle.status == "exact"
    return OracleCase(state=state, history=(), seed=7, oracle=oracle)


def test_unknown_stub_passes_without_creating_labels():
    summary = check_oracle_agreement([tactical_case()], threat_solver_stub)
    assert summary.checked_states == 1
    assert summary.abstained_states == 1
    assert summary.claimed_states == 0
    assert summary.checked_actions == 0


def test_harness_rejects_an_injected_false_positive():
    case = tactical_case()
    losing_action = next(action for action, value in case.oracle.action_values.items() if value != 1)

    def wrong_stub(state, node_cap, time_cap_ms):
        del state, node_cap, time_cap_ms
        return ThreatClaim(
            status="exact_partial",
            value=1,
            optimal_actions=(losing_action,),
            optimal_actions_complete=False,
            action_values={losing_action: 1},
            proof={"deliberately": "wrong"},
            method="vcf",
            unknown_reason=None,
        )

    with pytest.raises(AssertionError, match="false-positive"):
        check_oracle_agreement(
            [case], wrong_stub, proof_validator=lambda state, proof: True
        )


def test_harness_accepts_an_oracle_agreeing_claim_with_validated_proof():
    case = tactical_case()
    winning_action = next(action for action, value in case.oracle.action_values.items() if value == 1)

    def agreeing_stub(state, node_cap, time_cap_ms):
        del state, node_cap, time_cap_ms
        return ThreatClaim(
            status="exact_partial",
            value=1,
            optimal_actions=(winning_action,),
            optimal_actions_complete=False,
            action_values={winning_action: 1},
            proof={"fixture": "accepted"},
            method="vcf",
            unknown_reason=None,
        )

    summary = check_oracle_agreement(
        [case], agreeing_stub, proof_validator=lambda state, proof: True
    )
    assert summary.claimed_states == 1
    assert summary.checked_actions == 1


def test_legal_random_history_generator_is_deterministic():
    first = generate_oracle_cases(1, seed=19)
    second = generate_oracle_cases(1, seed=19)
    assert first[0].history == second[0].history
    assert np.array_equal(first[0].state.board, second[0].state.board)
    assert first[0].oracle.status == "exact"
