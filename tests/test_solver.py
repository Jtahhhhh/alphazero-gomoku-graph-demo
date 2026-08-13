import time

import numpy as np

from azgomoku.game import GomokuState
from azgomoku.solver import solve_state


def state(rows,to_play=1,k=3): return GomokuState(np.asarray(rows,dtype=np.int8),to_play=to_play,win_length=k)


def exhaustive(s, memo=None):
    memo={} if memo is None else memo
    if s.terminal(): return s.outcome_for(s.to_play)
    key=(s.board.tobytes(),s.to_play,s.win_length)
    if key not in memo: memo[key]=max(-exhaustive(s.play(int(a)),memo) for a in s.legal_actions())
    return memo[key]


def test_terminal_win_loss_and_draw_perspective():
    won=state([[1,1,1],[-1,-1,0],[0,0,0]],to_play=-1)
    assert solve_state(won).value==-1
    assert solve_state(GomokuState(won.board,to_play=1,win_length=3)).value==1
    draw=state([[1,-1,1],[1,-1,-1],[-1,1,1]],to_play=-1)
    assert solve_state(draw).value==0


def test_immediate_win_and_all_equal_optimal_actions():
    s=state([[1,-1,1,1,-1,-1],[0,1,1,0,1,-1],[-1,-1,-1,0,0,1],[-1,0,-1,1,1,-1],[1,0,-1,-1,1,-1],[1,-1,0,1,1,1]],to_play=-1,k=4)
    result=solve_state(s)
    assert result.status=="exact" and result.value==1
    assert result.optimal_actions==(6,15,32)
    assert set(result.action_values)==set(map(int,s.legal_actions()))


def test_limits_never_report_exact_and_are_deterministic():
    s=GomokuState.initial(3,3)
    assert solve_state(s,deadline_ms=0).status=="timeout"
    first=solve_state(s,node_budget=5); second=solve_state(s,node_budget=5)
    assert first.status==second.status=="node_budget" and first.nodes==second.nodes==5


def test_bounded_solver_matches_independent_exhaustive_enumeration():
    samples=[
        state([[1,1,0],[-1,0,0],[-1,0,0]]),
        state([[0,1,0],[-1,0,-1],[0,1,0]]),
        state([[1,-1,1],[-1,1,0],[0,0,-1]]),
        state([[1,-1,1],[1,-1,-1],[-1,1,0]]),
    ]
    for s in samples:
        result=solve_state(s)
        expected={int(a):-exhaustive(s.play(int(a))) for a in s.legal_actions()}
        assert result.status=="exact" and result.action_values==expected
        assert result.value==max(expected.values())
