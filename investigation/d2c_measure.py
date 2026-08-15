"""D2c measurement-only VCF coverage and unknown triage."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.game import GomokuState
from azgomoku.mcts import search
from azgomoku.solver import solve_actions
from azgomoku.vcf import solve_vcf
from models.rgat import RGAT


def load_resized_model(checkpoint: Path, size: int, hidden_dim=64, heads=4):
    model=RGAT(board_size=size,hidden_dim=hidden_dim,attention_heads=heads)
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    learned={key:value for key,value in saved.items() if not key.startswith("edge_")}
    missing,unexpected=model.load_state_dict(learned,strict=False)
    if unexpected or any(not key.startswith("edge_") for key in missing):
        raise ValueError(f"checkpoint resize mismatch: missing={missing}, unexpected={unexpected}")
    model.eval(); return model


def selfplay_population(model,size,k,games,sims,cap=500):
    states={}
    for seed in range(games):
        np.random.seed(seed); random.seed(seed); torch.manual_seed(seed)
        state=GomokuState.initial(size,k); ply=0
        while not state.terminal():
            states.setdefault(state_identifier(state),(state,ply))
            temperature=1.0 if ply<8 else 0.0
            pi=search(model,state,playouts=sims,temperature=temperature)
            action=int(np.random.choice(size*size,p=pi)); state=state.play(action); ply+=1
    values=list(states.values())
    if len(values)>cap:
        values=random.Random(12345).sample(values,cap)
    return values


def bucket(size,ply):
    bounds=(5,10) if size==6 else (10,25)
    if ply<bounds[0]: return f"0-{bounds[0]-1}"
    if ply<bounds[1]: return f"{bounds[0]}-{bounds[1]-1}"
    return f"{bounds[1]}+"


def signature(state):
    occupied=np.argwhere(state.board!=0)
    min_row,min_col=occupied.min(axis=0)
    stones=frozenset((int(r-min_row),int(c-min_col),int(state.board[r,c])) for r,c in occupied)
    return stones,int(state.to_play)


def bbox(state):
    occupied=np.argwhere(state.board!=0)
    if not len(occupied): return 0,0,0,0
    lo=occupied.min(axis=0); hi=occupied.max(axis=0)
    return int(hi[0]-lo[0]+1),int(hi[1]-lo[1]+1),int(lo[0]),int(lo[1])


def multiset_jaccard(left,right):
    intersection=sum((left & right).values()); union=sum((left | right).values())
    return intersection/union if union else 1.0


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def run(checkpoint,output,games6=50,games_large=20,sims6=64,sims_large=32,state_cap=500):
    checkpoint=Path(checkpoint); specs=((6,4,games6,sims6),(10,5,games_large,sims_large),(15,5,games_large,sims_large))
    cache_path=Path(output).with_suffix(".cache.pt")
    cache=torch.load(cache_path,map_location="cpu",weights_only=False) if cache_path.exists() else {"populations":{},"measurements":{}}
    populations={}; solved={}; coverage={}; by_ply={}; outcomes={}
    for size,k,games,sims in specs:
        started=time.perf_counter(); print(json.dumps({"stage":"board_start","size":size}),flush=True)
        if size not in cache["populations"]:
            model=load_resized_model(checkpoint,size)
            cache["populations"][size]=selfplay_population(model,size,k,games,sims,state_cap)
            torch.save(cache,cache_path); print(json.dumps({"stage":"population_done","size":size,"states":len(cache["populations"][size]),"seconds":time.perf_counter()-started}),flush=True)
        population=cache["populations"][size]; populations[size]=population
        if size not in cache["measurements"]:
            measured=[]
            for index,(state,ply) in enumerate(population):
                result=solve_vcf(state,node_cap=10_000,time_cap_ms=250)
                measured.append((result.status,result.unknown_reason))
                if (index+1)%50==0: print(json.dumps({"stage":"vcf_progress","size":size,"done":index+1,"total":len(population)}),flush=True)
            cache["measurements"][size]=measured; torch.save(cache,cache_path)
        entries=[]
        for (state,ply),(status,reason) in zip(population,cache["measurements"][size]):
            class Result: pass
            result=Result(); result.status=status; result.unknown_reason=reason; entries.append((state,ply,result))
        solved[size]=[(state,ply,result) for state,ply,result in entries if result.status=="exact_partial"]
        coverage[size]=len(solved[size])/len(entries)
        outcomes[size]=dict(Counter(result.unknown_reason or "proven" for _,_,result in entries))
        groups={}
        for state,ply,result in entries:
            key=bucket(size,ply); groups.setdefault(key,[]).append(result.status=="exact_partial")
        by_ply[size]={key:{"states":len(values),"coverage":sum(values)/len(values)} for key,values in groups.items()}
        print(json.dumps({"stage":"board_done","size":size,"coverage":coverage[size],"seconds":time.perf_counter()-started}),flush=True)

    triage=Counter(); exact_details=[]
    unknown6=[]
    for (state,ply),(status,reason) in zip(populations[6],cache["measurements"][6]):
        if status=="unknown":
            class Result: pass
            result=Result(); result.status=status; result.unknown_reason=reason; unknown6.append((state,ply,result))
    for index,(state,ply,result) in enumerate(unknown6):
        if result.unknown_reason=="budget":
            triage["C"]+=1; exact_details.append({"state_id":state_identifier(state),"ply":ply,"bucket":"C","exact_value":None}); continue
        print(json.dumps({"stage":"exact_start","index":index+1,"total":len(unknown6),"state_id":state_identifier(state),"ply":ply}),flush=True)
        exact=solve_actions(state)
        if exact.status!="exact": raise RuntimeError(f"unbounded exact solver did not finish for {state_identifier(state)}")
        group="A" if exact.value==1 else "B"; triage[group]+=1
        exact_details.append({"state_id":state_identifier(state),"ply":ply,"bucket":group,"exact_value":exact.value,"nodes":exact.nodes,"elapsed_ms":exact.elapsed_ms})
        print(json.dumps({"stage":"exact_done","index":index+1,"value":exact.value,"nodes":exact.nodes,"elapsed_ms":exact.elapsed_ms}),flush=True)
    u6=sum(triage.values()); p_a=triage["A"]/u6 if u6 else 0.; frac_c=triage["C"]/u6 if u6 else 0.

    sig10=Counter(signature(state) for state,_,_ in solved[10]); sig15=Counter(signature(state) for state,_,_ in solved[15]); jaccard=multiset_jaccard(sig10,sig15)
    boxes={size:[bbox(state) for state,_ in populations[size]] for size in (10,15)}
    medians={size:tuple(float(statistics.median(items[i] for items in boxes[size])) for i in range(4)) for size in (10,15)}
    dims_close=all(abs(medians[10][i]-medians[15][i])<=1 for i in (0,1))
    same_region=all(abs(medians[10][i]-medians[15][i])<=1 for i in (2,3))
    generator_independent=dims_close and same_region; artifact=jaccard>=.70 or generator_independent
    result={
        "inputs":{"checkpoint":str(checkpoint),"sha256":file_sha256(checkpoint),"mtime":checkpoint.stat().st_mtime,"specs":[{"size":s,"k":k,"games":g,"sims":m} for s,k,g,m in specs],"temperature":"1.0 for ply < 8, argmax after","seeds":f"0..games-1","state_cap":state_cap,"subsample_seed":12345,"vcf_budget":{"node_cap":10000,"time_cap_ms":250},"checkpoint_resize":"learned tensors reused; size-specific graph buffers regenerated"},
        "task1":{"population_sizes":{str(k):len(v) for k,v in populations.items()},"coverage_selfplay":{str(k):v for k,v in coverage.items()},"coverage_random":{"6":.9166666667,"10":.4166666667,"15":.4166666667},"by_ply":{str(k):v for k,v in by_ply.items()},"outcomes":{str(k):v for k,v in outcomes.items()}},
        "task2":{"A":triage["A"],"B":triage["B"],"C":triage["C"],"U6":u6,"P_A":p_a,"frac_C":frac_c,"details":exact_details},
        "task3":{"J":jaccard,"solved_signatures":{"10":sum(sig10.values()),"15":sum(sig15.values())},"bbox_medians":{"10":medians[10],"15":medians[15]},"generator_size_independent":generator_independent,"artifact":artifact},
    }
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(result,indent=2),encoding="utf-8"); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--games-6",type=int,default=50); parser.add_argument("--games-large",type=int,default=20); parser.add_argument("--sims-6",type=int,default=64); parser.add_argument("--sims-large",type=int,default=32); parser.add_argument("--state-cap",type=int,default=500); args=parser.parse_args()
    print(json.dumps(run(args.checkpoint,args.output,args.games_6,args.games_large,args.sims_6,args.sims_large,args.state_cap),indent=2))


if __name__=="__main__": main()
