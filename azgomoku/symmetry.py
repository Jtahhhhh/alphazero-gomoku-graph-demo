"""D4 coordinate transforms used by H1 generator deduplication."""

from __future__ import annotations

import numpy as np

from .game import GomokuState


RELATION_VECTORS={"horizontal":(0,1),"vertical":(1,0),"diagonal_down":(1,1),"diagonal_up":(1,-1)}


def transform_coord(row,col,size,symmetry):
    if symmetry not in range(8): raise ValueError("symmetry must be 0..7")
    turns=symmetry//2
    for _ in range(turns): row,col=size-1-col,row
    if symmetry%2: col=size-1-col
    return int(row),int(col)


def inverse_symmetry(symmetry,size=3):
    for candidate in range(8):
        if all(transform_coord(*transform_coord(r,c,size,symmetry),size,candidate)==(r,c) for r,c in ((0,0),(0,1),(1,0))): return candidate
    raise AssertionError("no inverse symmetry")


def transform_action(action,size,symmetry):
    row,col=divmod(int(action),size); row,col=transform_coord(row,col,size,symmetry); return row*size+col


def transform_board(board,symmetry):
    result=np.empty_like(board); size=board.shape[0]
    for row in range(size):
        for col in range(size): result[transform_coord(row,col,size,symmetry)]=board[row,col]
    return result


def transform_state(state,symmetry):
    last=-1 if state.last_move<0 else transform_action(state.last_move,state.size,symmetry)
    return GomokuState(transform_board(state.board,symmetry),state.to_play,last,state.win_length)


def transform_relation(relation,symmetry):
    dr,dc=RELATION_VECTORS[relation]
    origin=(1,1); target=(1+dr,1+dc); size=5
    a=transform_coord(*origin,size,symmetry); b=transform_coord(*target,size,symmetry)
    vector=(b[0]-a[0],b[1]-a[1]); normalized=(abs(vector[0]),0 if vector[1]==0 else (1 if vector[0]*vector[1]>=0 else -1))
    if vector[0]==0: normalized=(0,1)
    elif vector[1]==0: normalized=(1,0)
    return {value:key for key,value in RELATION_VECTORS.items()}[normalized]


def transform_flat_proof(proof,size,symmetry):
    result=dict(proof)
    result["action"]=transform_action(proof["action"],size,symmetry)
    result["critical_cells"]=sorted(transform_action(cell,size,symmetry) for cell in proof.get("critical_cells",[]))
    result["critical_relations"]=sorted({transform_relation(relation,symmetry) for relation in proof.get("critical_relations",[])})
    result["windows"]=sorted(sorted(transform_action(cell,size,symmetry) for cell in window) for window in proof.get("windows",[]))
    return result


def canonical_key(state):
    return min(
        (
            transform_board(state.board,symmetry).tobytes(),int(state.to_play),int(state.win_length),
            -1 if state.last_move<0 else transform_action(state.last_move,state.size,symmetry),
        )
        for symmetry in range(8)
    )


def d4_roundtrip_self_check():
    state=GomokuState(np.asarray([[1,0,-1],[0,1,0],[-1,0,0]],dtype=np.int8),to_play=1,last_move=4,win_length=3)
    proof={
        "action":1,"critical_cells":[0,1,2,3,4,5,6,7,8],
        "critical_relations":["diagonal_down","diagonal_up","horizontal","vertical"],
        "windows":[[0,1,2],[0,3,6],[0,4,8],[2,4,6]],
    }
    canonical=canonical_key(state)
    for symmetry in range(8):
        transformed=transform_state(state,symmetry); inverse=inverse_symmetry(symmetry,state.size)
        if canonical_key(transformed)!=canonical: return False
        mapped=transform_flat_proof(proof,state.size,symmetry)
        if transform_flat_proof(mapped,state.size,inverse)!=proof: return False
        actions=(1,5,7)
        if tuple(transform_action(transform_action(a,state.size,symmetry),state.size,inverse) for a in actions)!=actions: return False
    return True
