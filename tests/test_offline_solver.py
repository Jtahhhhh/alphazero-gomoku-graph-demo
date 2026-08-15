import numpy as np

from azgomoku.game import GomokuState
from azgomoku.offline_solver import solve_all_actions,solve_value,transform_action,transform_board
from azgomoku.solver import solve_actions


def fixture():
    return GomokuState(np.asarray([[1,-1,0],[0,1,-1],[0,0,0]],dtype=np.int8),to_play=1,win_length=3)


def transformed(state,symmetry):
    last=-1 if state.last_move<0 else transform_action(state.last_move,state.size,symmetry)
    return GomokuState(transform_board(state.board,symmetry).copy(),state.to_play,last,state.win_length)


def test_offline_solver_agrees_with_existing_solver_and_tt_toggle():
    state=fixture(); expected=solve_actions(state)
    with_tt=solve_all_actions(state,use_tt=True); without_tt=solve_all_actions(state,use_tt=False)
    assert with_tt.status==without_tt.status==expected.status=="exact"
    assert with_tt.value==without_tt.value==expected.value
    assert with_tt.optimal_actions==without_tt.optimal_actions==expected.optimal_actions
    assert with_tt.action_values==without_tt.action_values==expected.action_values


def test_all_d4_symmetries_preserve_value_and_map_optimal_actions():
    state=fixture(); base=solve_all_actions(state)
    for symmetry in range(8):
        result=solve_all_actions(transformed(state,symmetry))
        mapped=tuple(sorted(transform_action(action,state.size,symmetry) for action in base.optimal_actions))
        assert result.value==base.value and result.optimal_actions==mapped


def test_root_value_matches_full_width_value():
    state=fixture()
    assert solve_value(state).value==solve_all_actions(state).value


def test_root_value_two_pass_distinguishes_draw_and_loss():
    draw=GomokuState(np.asarray([[1,-1,1],[-1,1,-1],[-1,1,0]],dtype=np.int8),to_play=1,win_length=3)
    loss=GomokuState(np.asarray([[1,-1,0],[1,-1,0],[0,0,0]],dtype=np.int8),to_play=1,win_length=3)
    for state in (draw,loss):
        assert solve_value(state).value==solve_all_actions(state).value


def test_preferred_root_action_is_only_an_ordering_hint():
    state=fixture(); expected=solve_all_actions(state).value
    for action in map(int,state.legal_actions()):
        assert solve_value(state,preferred_action=action).value==expected
