"""Independent H1 solver validation entry point."""

import json
import time

from azgomoku.game import GomokuState
from azgomoku.solver import solve_state
from azgomoku.tactics import extract_tactical_proofs


def exhaustive(state,memo=None):
    memo={} if memo is None else memo
    if state.terminal(): return state.outcome_for(state.to_play)
    key=(state.board.tobytes(),state.to_play,state.win_length)
    if key not in memo: memo[key]=max(-exhaustive(state.play(int(a)),memo) for a in state.legal_actions())
    return memo[key]


def main():
    samples=[]
    base=GomokuState.initial(3,3)
    # Deterministic legal prefixes span opening, tactical, and near-terminal depths.
    histories=[(0,4),(0,4,1),(0,4,1,3),(0,4,1,3,8),(4,0,2,6),(0,1,2,4,3,5)]
    for history in histories:
        state=base
        for action in history:
            if state.terminal(): break
            state=state.play(action)
        if state.terminal(): continue
        start=time.perf_counter(); result=solve_state(state); expected=exhaustive(state); elapsed=(time.perf_counter()-start)*1000
        immediate=any("immediate_win" in proof["concepts"] for proof in extract_tactical_proofs(state))
        if immediate and result.value!=1: raise AssertionError("geometric immediate win did not solve to +1")
        if result.value!=expected: raise AssertionError(f"solver mismatch: {history}: {result.value} != {expected}")
        samples.append({"history":history,"value":result.value,"nodes":result.nodes,"elapsed_ms":elapsed,"immediate_certificate":immediate})
    print(json.dumps({"status":"PASS","independent_method":"exhaustive negamax without alpha-beta or production TT","samples":samples},indent=2))


if __name__=="__main__": main()
