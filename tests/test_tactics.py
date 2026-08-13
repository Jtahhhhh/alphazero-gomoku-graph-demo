import numpy as np

from azgomoku.game import GomokuState
from azgomoku.tactics import extract_tactical_proofs


def state(rows,to_play=1,k=4): return GomokuState(np.asarray(rows,dtype=np.int8),to_play=to_play,win_length=k)


def relation_for(rows,action,k=4):
    proofs=extract_tactical_proofs(state(rows,k=k))
    return {relation for proof in proofs if proof["action"]==action and "immediate_win" in proof["concepts"] for relation in proof["critical_relations"]}


def test_immediate_wins_in_all_four_directions():
    assert relation_for([[1,1,1,0],[0,-1,0,0],[0,-1,0,0],[0,-1,0,0]],3)=={"horizontal"}
    assert relation_for([[1,-1,0,0],[1,-1,0,0],[1,-1,0,0],[0,0,0,0]],12)=={"vertical"}
    assert relation_for([[1,-1,0,0],[0,1,-1,0],[0,0,1,-1],[0,0,0,0]],15)=={"diagonal_down"}
    assert relation_for([[0,0,-1,1],[0,-1,1,0],[-1,1,0,0],[0,0,0,0]],12)=={"diagonal_up"}


def test_mandatory_block_and_simple_fork():
    block=extract_tactical_proofs(state([[-1,-1,-1,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]))
    assert any(p["action"]==3 and p["concepts"]==["mandatory_block"] for p in block)
    fork=extract_tactical_proofs(state([[1,0,0],[0,1,-1],[0,-1,0]],k=3))
    proof=next(p for p in fork if p["action"]==3 and p["concepts"]==["simple_fork"])
    assert set(proof["critical_relations"])=={"diagonal_down","vertical"} and len(proof["windows"])>=2


def test_multiple_proofs_are_preserved():
    s=state([[1,1,1,0],[1,-1,-1,0],[1,-1,0,0],[0,0,0,0]])
    proofs=[p for p in extract_tactical_proofs(s) if p["action"] in (3,12) and "immediate_win" in p["concepts"]]
    assert len(proofs)>=2 and {p["action"] for p in proofs}>={3,12}
    assert all(p["action"] in s.legal_actions() for p in proofs)
