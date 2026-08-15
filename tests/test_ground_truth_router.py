import numpy as np

from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget,RouterStats,route_ground_truth
from azgomoku.solver import solve_actions
from azgomoku.vcf import replay_vcf_proof


def test_router_agrees_with_original_exact_solver_on_solvable_states():
    states=[
        GomokuState(np.asarray([[1,-1,1],[-1,1,0],[0,0,0]],dtype=np.int8),to_play=-1,win_length=3),
        GomokuState(np.asarray([[1,-1,1],[-1,1,-1],[-1,1,0]],dtype=np.int8),to_play=1,win_length=3),
    ]
    for state in states:
        exact=solve_actions(state)
        routed=route_ground_truth(state,GroundTruthBudget(100_000,5_000),empty_bounds={})
        assert routed.status=="exact_complete"
        assert routed.value==exact.value
        assert routed.optimal_actions==exact.optimal_actions
        assert routed.action_values==exact.action_values


def test_router_agrees_on_exact_solvable_6x6_and_prefers_complete():
    state=GomokuState(np.asarray([
        [1,-1,1,1,-1,-1],[0,1,1,0,1,-1],[-1,-1,-1,0,0,1],
        [-1,0,-1,1,1,-1],[1,0,-1,-1,1,-1],[1,-1,0,1,1,1],
    ],dtype=np.int8),to_play=-1,win_length=4)
    exact=solve_actions(state,deadline_ms=5_000,node_budget=1_000_000)
    routed=route_ground_truth(state,GroundTruthBudget(1_000_000,5_000))
    assert exact.status=="exact" and routed.status=="exact_complete"
    assert routed.value==exact.value and routed.optimal_actions==exact.optimal_actions and routed.action_values==exact.action_values


def test_router_partial_is_replay_verified_and_unknown_has_no_actions():
    winning=GomokuState(np.asarray([
        [0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,1,0,0,0],
        [0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],
    ],dtype=np.int8),to_play=1,win_length=4)
    partial=route_ground_truth(winning,GroundTruthBudget(100_000,2_000),empty_bounds={6:0})
    assert partial.status=="exact_partial" and replay_vcf_proof(winning,partial.proof)
    empty=GomokuState.initial(6,4)
    unknown=route_ground_truth(empty,GroundTruthBudget(100_000,2_000),empty_bounds={6:0})
    assert unknown.status=="unknown"
    assert unknown.optimal_actions is None and unknown.action_values is None and unknown.proof is None


def test_router_logs_complete_partial_unknown_rates():
    stats=RouterStats()
    complete_state=GomokuState(np.asarray([[1,-1,1],[-1,1,-1],[-1,1,0]],dtype=np.int8),to_play=1,win_length=3)
    route_ground_truth(complete_state,GroundTruthBudget(100_000,2_000),empty_bounds={},stats=stats)
    winning=GomokuState(np.asarray([[0]*6,[0]*6,[1,1,1,0,0,0],[0]*6,[0]*6,[0]*6],dtype=np.int8),to_play=1,win_length=4)
    route_ground_truth(winning,GroundTruthBudget(100_000,2_000),empty_bounds={6:0},stats=stats)
    route_ground_truth(GomokuState.initial(6,4),GroundTruthBudget(100_000,2_000),empty_bounds={6:0},stats=stats)
    assert stats.dict()["counts"]=={"exact_complete":1,"exact_partial":1,"unknown":1}
