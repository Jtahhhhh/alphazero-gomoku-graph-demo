"""Stronger offline exact solver for D2c triage and H1 ground truth."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .tactics import immediate_threats, windows


@dataclass(frozen=True)
class OfflineResult:
    status: str
    value: int | None
    optimal_actions: tuple[int, ...] | None
    action_values: dict[int, int] | None
    nodes: int
    elapsed_ms: float


class _Limit(RuntimeError):
    pass


def transform_board(board: np.ndarray, symmetry: int) -> np.ndarray:
    if symmetry not in range(8): raise ValueError("symmetry must be in range(8)")
    rotated=np.rot90(board,symmetry//2)
    return np.fliplr(rotated) if symmetry%2 else rotated


def transform_action(action: int, size: int, symmetry: int) -> int:
    marker=np.zeros((size,size),dtype=np.int8); marker[action//size,action%size]=1
    transformed=transform_board(marker,symmetry); return int(np.flatnonzero(transformed.reshape(-1))[0])


def canonical_board(board: np.ndarray) -> bytes:
    return min(transform_board(board,symmetry).tobytes() for symmetry in range(8))


class _Context:
    def __init__(self,time_cap_ms,node_cap,use_tt):
        self.start=time.perf_counter(); self.deadline=None if time_cap_ms is None else self.start+time_cap_ms/1000
        self.node_cap=node_cap; self.nodes=0; self.use_tt=use_tt; self.table={}

    def enter(self):
        if self.deadline is not None and time.perf_counter()>=self.deadline: raise _Limit
        if self.node_cap is not None and self.nodes>=self.node_cap: raise _Limit
        self.nodes+=1


def _key(state): return canonical_board(state.board),int(state.to_play),int(state.win_length)


@lru_cache(maxsize=None)
def _windows_by_cell(size,win_length):
    result=[[] for _ in range(size*size)]
    for _,cells in windows(size,win_length):
        for cell in cells: result[cell].append(cells)
    return tuple(tuple(group) for group in result)


def _creates_four_at(board,action,player,win_length):
    flat=board.reshape(-1)
    for cells in _windows_by_cell(board.shape[0],win_length)[action]:
        values=[int(flat[cell]) for cell in cells]
        if values.count(player)==win_length-1 and values.count(0)==1:
            return True
    return False


def _ordered_actions(state):
    actions=list(map(int,state.legal_actions())); player=int(state.to_play)
    wins=[]; blocks=[]; threats=[]; rest=[]
    opponent_completions={int(item[0]) for item in immediate_threats(state.board,-player,state.win_length)}
    center=(state.size-1)/2
    for action in actions:
        child=state.play(action)
        if child.winner()==player: wins.append(action)
        elif action in opponent_completions: blocks.append(action)
        # Any newly-created four must use the stone placed at ``action``.
        # Restricting the scan to those windows preserves the ordering class
        # while avoiding a full-board tactical extraction for every child.
        elif _creates_four_at(child.board,action,player,state.win_length): threats.append(action)
        else: rest.append(action)
    order=lambda action:(abs(action//state.size-center)+abs(action%state.size-center),action)
    return wins+sorted(blocks,key=order)+sorted(threats,key=order)+sorted(rest,key=order)


def _negamax(state,alpha,beta,ctx):
    ctx.enter()
    if state.terminal(): return state.outcome_for(state.to_play)
    key=_key(state); alpha_original=alpha; beta_original=beta
    if ctx.use_tt and key in ctx.table:
        cached,flag=ctx.table[key]
        if flag=="exact": return cached
        if flag=="lower": alpha=max(alpha,cached)
        else: beta=min(beta,cached)
        if alpha>=beta: return cached
    best=-2
    for action in _ordered_actions(state):
        value=-_negamax(state.play(action),-beta,-alpha,ctx)
        if value>best: best=value
        if value>alpha: alpha=value
        if alpha>=beta: break
    if ctx.use_tt:
        flag="upper" if best<=alpha_original else ("lower" if best>=beta_original else "exact")
        ctx.table[key]=(best,flag)
    return best


def solve_value(state,time_cap_ms=None,node_cap=None,use_tt=True,preferred_action=None):
    ctx=_Context(time_cap_ms,node_cap,use_tt)
    try:
        if preferred_action is not None:
            preferred_action=int(preferred_action)
            if preferred_action not in set(map(int,state.legal_actions())):
                raise ValueError("preferred_action must be legal")
            # A preferred root action is ordering information only. Its child
            # is still searched exactly, and an early return requires negamax
            # itself to prove that this action forces +1.
            preferred_value=-_negamax(state.play(preferred_action),-1,0,ctx)
            if preferred_value==1:
                return OfflineResult("exact",1,None,None,ctx.nodes,(time.perf_counter()-ctx.start)*1000)
        # The value domain is {-1, 0, +1}.  A zero-window pass answers the
        # triage question ("is this a forced win?") with substantially more
        # pruning.  Only fail-low positions need a second pass to distinguish
        # draw from loss; together the two passes still return the exact value.
        value=_negamax(state,0,1,ctx)
        if value != 1:
            value=_negamax(state,-1,0,ctx)
        status="exact"
    except _Limit: value=None; status="unknown"
    return OfflineResult(status,value,None,None,ctx.nodes,(time.perf_counter()-ctx.start)*1000)


def solve_all_actions(state,time_cap_ms=None,node_cap=None,use_tt=True):
    ctx=_Context(time_cap_ms,node_cap,use_tt); values={}
    try:
        if state.terminal(): return OfflineResult("exact",state.outcome_for(state.to_play),(),{},0,0.)
        for action in _ordered_actions(state): values[action]=-_negamax(state.play(action),-1,1,ctx)
        value=max(values.values()); optimal=tuple(sorted(action for action,candidate in values.items() if candidate==value)); status="exact"
    except _Limit: value=None; optimal=None; values=None; status="unknown"
    return OfflineResult(status,value,optimal,values,ctx.nodes,(time.perf_counter()-ctx.start)*1000)
