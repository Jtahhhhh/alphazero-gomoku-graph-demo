"""Generate replay-verified schema-v2 H1 candidates from mid/late game states."""

from __future__ import annotations

import argparse,hashlib,json,random
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch

from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget,RouterStats,route_ground_truth
from azgomoku.h1_schema import make_record,validate_record
from azgomoku.mcts import search
from azgomoku.explanation.game_export import select_action
from azgomoku.symmetry import canonical_key,d4_roundtrip_self_check
from models.rgat import RGAT
from models.rgcn import RGCN


GENERATOR_VERSION="h1_selfplay_data_v2"
# D2c buckets: exclude sparse opening, retain both mid and late populations.
DEFAULT_SPECS=((6,4,5,20),(10,5,10,40),(15,5,10,40))
MODELS={"rgat":RGAT,"rgcn":RGCN}


def replay(history,size=6,k=4):
    state=GomokuState.initial(size,k)
    for action in history:
        if state.terminal(): raise ValueError("history continues after terminal state")
        state=state.play(action)
    return state


def ply_bucket(size,ply):
    if size==6: return "10+" if ply>=10 else ("5-9" if ply>=5 else "0-4")
    return "25+" if ply>=25 else ("10-24" if ply>=10 else "0-9")


def _sample_history(rng,size,k,min_ply,max_ply):
    state=GomokuState.initial(size,k); history=[]; target=rng.randint(min_ply,max_ply); fallback=None
    while len(history)<target and not state.terminal():
        if len(history)>=min_ply: fallback=(state,list(history))
        # mode=data semantics: stochastic sampling from a legal policy.  The
        # generator keeps this injectable/lightweight; a trained-model source
        # can replace the uniform policy without changing routing or schema.
        action=rng.choice(list(map(int,state.legal_actions())))
        history.append(action); state=state.play(action)
    if not state.terminal() and len(history)>=min_ply: return state,history
    return fallback if fallback is not None else (state,history)


class ModelSelfPlayDataSource:
    """Generate states with the same stochastic `mode=data` move selection as arena export."""
    def __init__(self,checkpoint,model_type="rgat",mcts_playouts=32,opening_temperature_moves=10,temperature=1.,late_temperature=0.,device="cpu"):
        self.checkpoint=Path(checkpoint); self.model_type=model_type; self.mcts_playouts=mcts_playouts
        self.opening_temperature_moves=opening_temperature_moves; self.temperature=temperature; self.late_temperature=late_temperature; self.models={}
        if model_type not in MODELS: raise ValueError("model_type must be rgat or rgcn")
        if device=="auto": device="cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available(): raise ValueError("CUDA requested but torch.cuda.is_available() is false")
        self.device=torch.device(device)
        digest=hashlib.sha256(); digest.update(self.checkpoint.read_bytes()); self.checkpoint_sha256=digest.hexdigest()

    def _model(self,size):
        if size not in self.models:
            model=MODELS[self.model_type](board_size=size)
            saved=torch.load(self.checkpoint,map_location="cpu",weights_only=True)
            learned={key:value for key,value in saved.items() if not key.startswith("edge_")}
            missing,unexpected=model.load_state_dict(learned,strict=False)
            if unexpected or any(not key.startswith("edge_") for key in missing): raise ValueError(f"checkpoint resize mismatch: {missing}, {unexpected}")
            model.to(self.device).eval(); self.models[size]=model
        return self.models[size]

    def __call__(self,rng,size,k,min_ply,max_ply):
        target=rng.randint(min_ply,max_ply)
        state,history=self.trajectory(rng,size,k,target)
        if not state.terminal() and len(history)>=min_ply: return state,history
        # ``trajectory`` returns the latest non-terminal prefix when the game
        # ends before target, so this branch only covers a game before min_ply.
        return state,history

    def trajectory(self,rng,size,k,target):
        state=GomokuState.initial(size,k); history=[]; model=self._model(size); fallback=None
        np_rng=np.random.default_rng(rng.getrandbits(64))
        while len(history)<target and not state.terminal():
            fallback=(state,list(history))
            _,root=search(model,state,playouts=self.mcts_playouts,temperature=1.,return_root=True)
            selection_temperature=self.temperature if len(history)<=self.opening_temperature_moves else self.late_temperature
            action=select_action(root,size*size,mode="data",temperature=selection_temperature,rng=np_rng)
            history.append(action); state=state.play(action)
        if not state.terminal(): return state,history
        # A sampled target can lie beyond the actual game length.  Preserve the
        # latest non-terminal state in the requested mid/late bucket instead of
        # discarding the whole self-play trajectory.
        return fallback if fallback is not None else (state,history)

    def provenance(self):
        return {"state_source":"selfplay_mode_data","model_type":self.model_type,"checkpoint":str(self.checkpoint),"checkpoint_sha256":self.checkpoint_sha256,"mcts_playouts":self.mcts_playouts,"temperature":self.temperature,"opening_temperature_moves":self.opening_temperature_moves,"late_temperature":self.late_temperature,"device":str(self.device)}


def record(state,history,result,seed,*,dedup_mode="d4",generator_version=GENERATOR_VERSION,provenance_extra=None):
    return make_record(state,history,result,seed,generator_version=generator_version,ply=len(history),dedup_mode=dedup_mode,provenance_extra=provenance_extra)


def generate(
    target=24,seed=7,attempts=30_000,deadline_ms=2_000,node_cap=1_000_000,
    specs=DEFAULT_SPECS,include_unknown=True,state_source=None,benchmark_always_try_exact=True,
):
    rng=random.Random(seed); accepted=[]; seen=set(); stats=Counter(); by_group=defaultdict(Counter); router_stats=RouterStats()
    dedup_mode="d4" if d4_roundtrip_self_check() else "state_id"
    per_board=max(1,(target+len(specs)-1)//len(specs))
    accepted_by_size=Counter()
    budget=GroundTruthBudget(node_cap,time_cap_ms=deadline_ms)
    for attempt in range(attempts):
        if all(accepted_by_size[size]>=per_board for size,_,_,_ in specs) or len(accepted)>=target: break
        size,k,min_ply,max_ply=specs[attempt%len(specs)]
        if accepted_by_size[size]>=per_board: continue
        stats["attempted"]+=1; state,history=(state_source or _sample_history)(rng,size,k,min_ply,max_ply)
        if state.terminal(): stats["rejected_terminal"]+=1; continue
        if len(history)<min_ply: stats["rejected_before_midlate"]+=1; continue
        rebuilt=replay(history,size,k)
        if not np.array_equal(rebuilt.board,state.board) or rebuilt.to_play!=state.to_play: raise AssertionError("history replay mismatch")
        key=canonical_key(state) if dedup_mode=="d4" else (state.board.tobytes(),int(state.to_play),int(state.last_move))
        if key in seen: stats["rejected_duplicate"]+=1; continue
        # Benchmark generation is a one-time offline job: always spend its
        # calibrated budget on the verified exact solver before VCF. Runtime
        # routing retains E* as a latency optimization.
        result=route_ground_truth(state,budget,empty_bounds={} if benchmark_always_try_exact else None,stats=router_stats)
        group=by_group[(size,ply_bucket(size,len(history)))]; group[result.status]+=1
        if result.status=="unknown" and not include_unknown: stats["unknown_not_written"]+=1; continue
        source_metadata=state_source.provenance() if state_source is not None and hasattr(state_source,"provenance") else {"state_source":"test_uniform_data_policy"}
        source_metadata={**source_metadata,"benchmark_exact_skip_policy":"disabled_always_try_verified_exact" if benchmark_always_try_exact else "runtime_E_star"}
        item=record(state,history,result,seed,dedup_mode=dedup_mode,provenance_extra=source_metadata)
        validation=validate_record(item)
        if not validation.accepted: raise AssertionError(f"writer produced invalid record: {validation.errors}")
        seen.add(key); accepted.append(item); accepted_by_size[size]+=1; stats["accepted"]+=1
        stats["ground_truth_denominator"]+=int(validation.eligible)
    summary={
        "generator_version":GENERATOR_VERSION,"seed":seed,"budget":{"node_cap":node_cap,"time_cap_ms":deadline_ms},
        "dedup_mode":dedup_mode,"d4_roundtrip_passed":dedup_mode=="d4","counts":dict(stats),
        "router":router_stats.dict(),"by_board_and_ply":{
            f"{size}:{bucket}":{"counts":dict(counts),"total":sum(counts.values())}
            for (size,bucket),counts in sorted(by_group.items())
        },
    }
    return accepted,summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--target",type=int,default=24); parser.add_argument("--seed",type=int,default=7); parser.add_argument("--attempts",type=int,default=30_000); parser.add_argument("--deadline-ms",type=int,default=2_000); parser.add_argument("--node-cap",type=int,default=1_000_000); parser.add_argument("--exclude-unknown",action="store_true"); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--model-type",choices=tuple(MODELS),default="rgat"); parser.add_argument("--mcts-playouts",type=int,default=32); parser.add_argument("--output",type=Path,default=Path("diagnostic/h1_candidates_v2.jsonl")); args=parser.parse_args()
    source=ModelSelfPlayDataSource(args.checkpoint,args.model_type,args.mcts_playouts)
    records,summary=generate(args.target,args.seed,args.attempts,args.deadline_ms,args.node_cap,include_unknown=not args.exclude_unknown,state_source=source)
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text("".join(json.dumps(item,separators=(",",":"))+"\n" for item in records),encoding="utf-8")
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"records":len(records),**summary},indent=2))
    if len(records)<args.target: raise SystemExit(f"generated only {len(records)} of {args.target} requested states")


if __name__=="__main__": main()
