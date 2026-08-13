"""Generate a small solver-certified H1 benchmark from legal move histories."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.game import GomokuState
from azgomoku.solver import solve_state
from azgomoku.tactics import extract_tactical_proofs


CONCEPTS=("immediate_win","mandatory_block","simple_fork")


def replay(history,size=6,k=4):
    state=GomokuState.initial(size,k)
    for action in history:
        if state.terminal(): raise ValueError("history continues after terminal state")
        state=state.play(action)
    return state


def transforms(board,last_move):
    n=board.shape[0]; row,col=(-1,-1) if last_move<0 else divmod(last_move,n)
    for rotations in range(4):
        rotated=np.rot90(board,rotations)
        if last_move>=0:
            rr,cc=row,col
            for _ in range(rotations): rr,cc=n-1-cc,rr
            transformed_last=rr*n+cc
        else: transformed_last=-1
        yield rotated,transformed_last
        flipped=np.fliplr(rotated)
        yield flipped, (-1 if transformed_last<0 else (transformed_last//n)*n+(n-1-transformed_last%n))


def canonical_key(state):
    return min((board.tobytes(),int(state.to_play),int(last)) for board,last in transforms(state.board,state.last_move))


def record(state,history,result,proofs,seed):
    return {
        "schema_version":1,
        "state_id":state_identifier(state),
        "state":{"board_size":state.size,"win_length":state.win_length,"current_player":int(state.to_play),"last_move":int(state.last_move),"board":state.board.astype(int).tolist(),"legal_actions":[int(a) for a in state.legal_actions()]},
        "provenance":{"generator":"legal_random_history_v1","seed":seed,"history":[int(a) for a in history]},
        "solver":result.dict(),
        "valid_proofs":proofs,
    }


def generate(target=24,seed=7,attempts=30000,deadline_ms=2000):
    rng=random.Random(seed); accepted=[]; seen=set(); stats=Counter(); concept_counts=Counter(); relation_counts=Counter()
    for _ in range(attempts):
        stats["attempted"]+=1; state=GomokuState.initial(); history=[]
        desired=rng.randint(26,32)
        while len(history)<desired and not state.terminal():
            action=rng.choice(list(map(int,state.legal_actions()))); history.append(action); state=state.play(action)
        if state.terminal(): stats["rejected_terminal"]+=1; continue
        assert np.count_nonzero(state.board==1) in (np.count_nonzero(state.board==-1),np.count_nonzero(state.board==-1)+1)
        assert replay(history).board.tolist()==state.board.tolist() and replay(history).last_move==state.last_move
        key=canonical_key(state)
        if key in seen: stats["rejected_duplicate"]+=1; continue
        proofs=extract_tactical_proofs(state)
        if not proofs: stats["rejected_no_proof"]+=1; continue
        result=solve_state(state,deadline_ms=deadline_ms,node_budget=1_000_000)
        if result.status!="exact": stats["solver_"+result.status]+=1; continue
        proofs=[proof for proof in proofs if proof["action"] in result.optimal_actions]
        if not proofs: stats["rejected_no_optimal_proof"]+=1; continue
        concepts={concept for proof in proofs for concept in proof["concepts"] if concept in CONCEPTS}
        relations={relation for proof in proofs for relation in proof["critical_relations"]}
        # Soft balancing: do not add oversupplied examples while a class remains absent.
        if accepted and all(concept_counts[c]>=4 for c in CONCEPTS) and all(relation_counts[r]>=4 for r in ("horizontal","vertical","diagonal_down","diagonal_up")) and len(accepted)>=target: break
        score=sum(concept_counts[c] for c in concepts)+sum(relation_counts[r] for r in relations)
        if len(accepted)>=target and score>min(concept_counts.values(),default=0)+min(relation_counts.values(),default=0)+4: continue
        seen.add(key); accepted.append(record(state,history,result,proofs,seed)); stats["accepted"]+=1
        concept_counts.update(concepts); relation_counts.update(relations)
        if len(accepted)>=target and all(concept_counts[c]>=3 for c in CONCEPTS) and all(relation_counts[r]>=3 for r in ("horizontal","vertical","diagonal_down","diagonal_up")): break
    stats["multiple_optimal_states"]=sum(len(x["solver"]["optimal_actions"])>1 for x in accepted)
    stats["multiple_proof_states"]=sum(len(x["valid_proofs"])>1 for x in accepted)
    return accepted,{"counts":dict(stats),"concepts":dict(concept_counts),"relations":dict(relation_counts)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--target",type=int,default=24); parser.add_argument("--seed",type=int,default=7); parser.add_argument("--attempts",type=int,default=30000); parser.add_argument("--deadline-ms",type=int,default=2000); parser.add_argument("--output",type=Path,default=Path("diagnostic/h1_tactical.jsonl")); args=parser.parse_args()
    records,summary=generate(args.target,args.seed,args.attempts,args.deadline_ms); args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text("".join(json.dumps(record,separators=(",",":"))+"\n" for record in records),encoding="utf-8")
    summary_path=args.output.with_suffix(".summary.json"); summary_path.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"records":len(records),**summary},indent=2))
    if len(records)<args.target: raise SystemExit(f"generated only {len(records)} of {args.target} requested states")


if __name__=="__main__": main()
