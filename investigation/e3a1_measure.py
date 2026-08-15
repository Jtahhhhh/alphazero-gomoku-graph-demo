"""E-3a.1 budget calibration and production label-distribution measurement.

The measurement is resumable.  Benchmark generation always passes
``empty_bounds={}`` to the router, so the original verified exact solver is
tried before VCF regardless of the runtime E* optimization.
"""

from __future__ import annotations

import argparse,csv,json,random,time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np
import torch

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget,route_ground_truth
from azgomoku.h1_schema import make_record,validate_record
from azgomoku.solver import solve_actions
from azgomoku.symmetry import canonical_key,d4_roundtrip_self_check
from investigation.generate_h1_benchmark import ModelSelfPlayDataSource,ply_bucket


SPECS={6:(4,5,35),10:(5,10,60),15:(5,10,60)}
GENERATOR_VERSION="h1_selfplay_data_v2_budget_calibration"


def _save_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding="utf-8")


def _game_seed(base_seed,size,game):
    return int(np.random.SeedSequence([base_seed,size,game]).generate_state(1,dtype=np.uint32)[0])


def collect_population(source,output,target_per_board,base_seed,max_games=500):
    """Collect every mid/late state along deterministic mode=data trajectories."""
    output=Path(output)
    cache=torch.load(output,map_location="cpu",weights_only=False) if output.exists() else {
        "metadata":{"base_seed":base_seed,"target_per_board":target_per_board,"source":source.provenance(),"dedup_mode":"d4"},
        "states":{size:[] for size in SPECS},"games":{size:0 for size in SPECS},
    }
    if not d4_roundtrip_self_check(): raise RuntimeError("D4 round-trip gate failed")
    for size,(k,min_ply,max_ply) in SPECS.items():
        seen={canonical_key(item["state"]) for item in cache["states"][size]}
        while len(cache["states"][size])<target_per_board and cache["games"][size]<max_games:
            game_index=cache["games"][size]; seed=_game_seed(base_seed,size,game_index); rng=random.Random(seed)
            # Ask the source for one full trajectory target and reconstruct all
            # prefixes.  Its selection is the same mode=data path used by E-3a.
            terminal,history=source.trajectory(rng,size,k,max_ply)
            state=GomokuState.initial(size,k); added=0
            for ply,action in enumerate(history):
                if ply>=min_ply and not state.terminal():
                    key=canonical_key(state)
                    if key not in seen:
                        seen.add(key); cache["states"][size].append({"state":state,"history":tuple(history[:ply]),"ply":ply,"game_seed":seed,"game_index":game_index}); added+=1
                        if len(cache["states"][size])>=target_per_board: break
                state=state.play(action)
            cache["games"][size]+=1; output.parent.mkdir(parents=True,exist_ok=True); torch.save(cache,output)
            print(json.dumps({"stage":"population","board":size,"game":game_index,"added":added,"states":len(cache["states"][size]),"target":target_per_board}),flush=True)
        if len(cache["states"][size])<target_per_board: raise RuntimeError(f"only {len(cache['states'][size])}/{target_per_board} states for board {size}")
    return cache


def _stratified_sample(items,count,seed):
    groups=defaultdict(list)
    for item in items: groups[ply_bucket(item["state"].size,item["ply"])].append(item)
    rng=random.Random(seed); selected=[]; selected_ids=set()
    while len(selected)<min(count,len(items)):
        progressed=False
        for bucket in sorted(groups):
            candidates=[item for item in groups[bucket] if state_identifier(item["state"]) not in selected_ids]
            if candidates and len(selected)<count:
                choice=rng.choice(candidates); selected.append(choice); selected_ids.add(state_identifier(choice["state"])); progressed=True
        if not progressed: break
    return selected


def calibrate(cache,output_dir,budgets_ms,sample_per_board,node_cap):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); progress_path=output_dir/"calibration_progress.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    selected={size:_stratified_sample(cache["states"][size],sample_per_board,7100+size) for size in SPECS}
    for size,items in selected.items():
        for item in items:
            state=item["state"]; state_id=state_identifier(state)
            for budget_ms in budgets_ms:
                key=f"{size}:{state_id}:{budget_ms}"
                if key in progress: continue
                result=solve_actions(state,deadline_ms=budget_ms,node_budget=node_cap)
                progress[key]={"board_size":size,"state_id":state_id,"ply":item["ply"],"ply_bucket":ply_bucket(size,item["ply"]),"empty_count":len(state.legal_actions()),"budget_ms":budget_ms,"status":result.status,"value":result.value,"nodes":result.nodes,"elapsed_ms":result.elapsed_ms}
                _save_json(progress_path,progress)
                print(json.dumps({"stage":"calibration","board":size,"budget_ms":budget_ms,"status":result.status,"elapsed_ms":result.elapsed_ms}),flush=True)
    rows=list(progress.values()); summary={"sample_per_board":sample_per_board,"node_cap":node_cap,"budgets_ms":budgets_ms,"boards":{}}
    chosen={}
    for size in SPECS:
        rates=[]
        for budget in budgets_ms:
            group=[row for row in rows if row["board_size"]==size and row["budget_ms"]==budget]
            rates.append({"budget_ms":budget,"states":len(group),"exact_complete":sum(row["status"]=="exact" for row in group),"rate":sum(row["status"]=="exact" for row in group)/len(group),"wall_time_s":sum(row["elapsed_ms"] for row in group)/1000,"median_elapsed_ms":float(np.median([row["elapsed_ms"] for row in group]))})
        # Pick the earliest measured point after which the next step gains <5pp.
        choice=budgets_ms[-1]
        for current,nxt in zip(rates,rates[1:]):
            if nxt["rate"]-current["rate"]<.05: choice=current["budget_ms"]; break
        chosen[size]=choice; summary["boards"][str(size)]={"rates":rates,"BUDGET_star_ms":choice}
        selected_rows=[row for row in rows if row["board_size"]==size and row["budget_ms"]==choice]
        runtime_bound=15 if size==6 else None
        skip_baseline=sum(row["status"]=="exact" and (runtime_bound is None or row["empty_count"]<=runtime_bound) for row in selected_rows)
        summary["boards"][str(size)]["runtime_skip_baseline_complete_rate"]=skip_baseline/len(selected_rows)
        summary["boards"][str(size)]["benchmark_always_try_complete_rate"]=sum(row["status"]=="exact" for row in selected_rows)/len(selected_rows)
    summary["BUDGET_star_ms"]={str(k):v for k,v in chosen.items()}; _save_json(output_dir/"calibration_summary.json",summary)
    with (output_dir/"complete_rate_vs_budget.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=("board_size","budget_ms","states","exact_complete","rate","wall_time_s","median_elapsed_ms")); writer.writeheader()
        for size,data in summary["boards"].items():
            for row in data["rates"]: writer.writerow({"board_size":size,**row})
    (output_dir/"complete_rate_vs_budget.svg").write_text(render_budget_svg(summary),encoding="utf-8")
    return summary


def render_budget_svg(summary):
    width,height=850,480; left,right,top,bottom=70,35,45,420; budgets=summary["budgets_ms"]; colors={"6":"#2563eb","10":"#16a34a","15":"#dc2626"}
    x=lambda index:left+index/max(1,len(budgets)-1)*(width-left-right); y=lambda rate:bottom-rate*(bottom-top)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/><text x="{width/2}" y="26" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Exact-complete rate vs budget</text>']
    for pct in (0,.25,.5,.75,1): parts.append(f'<line x1="{left}" y1="{y(pct)}" x2="{width-right}" y2="{y(pct)}" stroke="#e2e8f0"/><text x="{left-8}" y="{y(pct)+4}" text-anchor="end" font-family="Arial" font-size="11">{pct:.0%}</text>')
    for index,budget in enumerate(budgets): parts.append(f'<text x="{x(index)}" y="{bottom+22}" text-anchor="middle" font-family="Arial" font-size="11">{budget/1000:g}s</text>')
    for size,data in summary["boards"].items():
        points=" ".join(f'{x(i)},{y(row["rate"])}' for i,row in enumerate(data["rates"])); parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[size]}" stroke-width="3"/>')
        for i,row in enumerate(data["rates"]): parts.append(f'<circle cx="{x(i)}" cy="{y(row["rate"])}" r="5" fill="{colors[size]}"/>')
        parts.append(f'<text x="{width-right-55}" y="{55+20*list(summary["boards"]).index(size)}" fill="{colors[size]}" font-family="Arial" font-size="12">{size}x{size}</text>')
    return "".join(parts)+"</svg>"


def route_production(cache,output_dir,budgets,node_cap,target_per_board):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); progress_path=output_dir/"routing_progress.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    records_path=output_dir/"production_candidates.jsonl"
    expected_budget={size:int(budgets[str(size)]) for size in SPECS}
    stale={key for key,row in progress.items() if int(row.get("budget_ms",500))!=expected_budget[row["board_size"]]}
    if stale:
        stale_pairs={(progress[key]["board_size"],progress[key]["state_id"]) for key in stale}
        for key in stale: del progress[key]
        _save_json(progress_path,progress)
        if records_path.exists():
            retained=[]
            for line in records_path.read_text(encoding="utf-8").splitlines():
                record=json.loads(line); pair=(int(record["state"]["board_size"]),record["state_id"])
                if pair not in stale_pairs: retained.append(line)
            records_path.write_text("".join(line+"\n" for line in retained),encoding="utf-8")
        print(json.dumps({"stage":"invalidate_stale_budget","records":len(stale),"boards":sorted({pair[0] for pair in stale_pairs})}),flush=True)
    for size in SPECS:
        for index,item in enumerate(cache["states"][size][:target_per_board]):
            state=item["state"]; state_id=state_identifier(state); key=f"{size}:{state_id}"
            if key in progress: continue
            started=time.perf_counter(); budget=GroundTruthBudget(node_cap,int(budgets[str(size)]))
            # Benchmark-only policy: always try verified exact first. Runtime
            # router defaults (including E*=15) remain unchanged.
            result=route_ground_truth(state,budget,empty_bounds={})
            record=make_record(state,item["history"],result,item["game_seed"],generator_version=GENERATOR_VERSION,ply=item["ply"],dedup_mode="d4",provenance_extra={**cache["metadata"]["source"],"game_index":item["game_index"],"benchmark_exact_skip_policy":"disabled_always_try_verified_exact"})
            validation=validate_record(record)
            if not validation.accepted: raise RuntimeError(f"invalid writer output {key}: {validation.errors}")
            with records_path.open("a",encoding="utf-8") as handle: handle.write(json.dumps(record,separators=(",",":"))+"\n")
            progress[key]={"board_size":size,"state_id":state_id,"ply":item["ply"],"ply_bucket":ply_bucket(size,item["ply"]),"empty_count":len(state.legal_actions()),"budget_ms":budget.time_cap_ms,"status":result.status,"method":result.method,"eligible":validation.eligible,"nodes":result.nodes,"solver_elapsed_ms":result.elapsed_ms,"wall_elapsed_ms":(time.perf_counter()-started)*1000,"partial_replay_pass":result.status!="exact_partial" or validation.accepted}
            _save_json(progress_path,progress)
            print(json.dumps({"stage":"routing","board":size,"done":index+1,"target":target_per_board,"status":result.status,"wall_ms":progress[key]["wall_elapsed_ms"]}),flush=True)
    return summarize_production(list(progress.values()),output_dir,target_per_board,budgets)


def summarize_production(rows,output_dir,target_per_board,budgets):
    distribution=defaultdict(Counter); board_counts=defaultdict(Counter); complete_ply=defaultdict(Counter)
    for row in rows:
        distribution[(row["board_size"],row["ply_bucket"])][row["status"]]+=1; board_counts[row["board_size"]][row["status"]]+=1
        if row["status"]=="exact_complete": complete_ply[row["board_size"]][row["ply_bucket"]]+=1
    summary={"target_per_board":target_per_board,"BUDGET_star_ms":budgets,"distribution":{},"boards":{},"complete_phase_distribution":{},"complete_ply_stats":{},"wall_time_total_s":sum(row["wall_elapsed_ms"] for row in rows)/1000}
    for (size,bucket),counts in sorted(distribution.items()):
        summary["distribution"][f"{size}:{bucket}"]={"complete":counts["exact_complete"],"partial":counts["exact_partial"],"unknown":counts["unknown"],"total":sum(counts.values())}
    for size,counts in sorted(board_counts.items()):
        group=[row for row in rows if row["board_size"]==size]; total=len(group)
        summary["boards"][str(size)]={"complete":counts["exact_complete"],"partial":counts["exact_partial"],"unknown":counts["unknown"],"total":total,"complete_rate":counts["exact_complete"]/total,"partial_rate":counts["exact_partial"]/total,"unknown_rate":counts["unknown"]/total,"mean_wall_ms":float(np.mean([row["wall_elapsed_ms"] for row in group])),"median_wall_ms":float(np.median([row["wall_elapsed_ms"] for row in group])),"partial_replay_pass":sum(row["status"]=="exact_partial" and row["partial_replay_pass"] for row in group),"partial_total":counts["exact_partial"]}
        summary["complete_phase_distribution"][str(size)]=dict(complete_ply[size])
        complete_plies=[row["ply"] for row in group if row["status"]=="exact_complete"]
        summary["complete_ply_stats"][str(size)]=None if not complete_plies else {"count":len(complete_plies),"min":min(complete_plies),"median":float(np.median(complete_plies)),"mean":float(np.mean(complete_plies)),"max":max(complete_plies)}
    _save_json(Path(output_dir)/"production_summary.json",summary); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--target-per-board",type=int,default=300); parser.add_argument("--seed",type=int,default=7); parser.add_argument("--mcts-playouts",type=int,default=8); parser.add_argument("--calibration-sample",type=int,default=8); parser.add_argument("--budgets-ms",type=int,nargs="+",default=(500,2000,8000)); parser.add_argument("--node-cap",type=int,default=1_000_000); parser.add_argument("--stage",choices=("population","calibrate","route","all"),default="all"); args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True); population_path=args.output_dir/"population.pt"; source=ModelSelfPlayDataSource(args.checkpoint,"rgat",args.mcts_playouts)
    if args.stage in ("population","all"): cache=collect_population(source,population_path,args.target_per_board,args.seed)
    else: cache=torch.load(population_path,map_location="cpu",weights_only=False)
    if args.stage=="population": return
    if args.stage in ("calibrate","all"): calibration=calibrate(cache,args.output_dir,args.budgets_ms,args.calibration_sample,args.node_cap)
    else: calibration=json.loads((args.output_dir/"calibration_summary.json").read_text())
    if args.stage=="calibrate": return
    if args.stage in ("route","all"): route_production(cache,args.output_dir,calibration["BUDGET_star_ms"],args.node_cap,args.target_per_board)


if __name__=="__main__": main()
