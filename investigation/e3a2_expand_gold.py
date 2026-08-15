"""Expand the 6x6 mid-game exact-complete gold set before E-3b."""

from __future__ import annotations

import argparse,json,random,time
from collections import Counter
from pathlib import Path

import numpy as np

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget,route_ground_truth
from azgomoku.h1_schema import make_record,state_from_record,validate_record
from azgomoku.offline_solver import solve_all_actions
from azgomoku.solver import solve_actions
from azgomoku.symmetry import canonical_key,d4_roundtrip_self_check
from investigation.generate_h1_benchmark import ModelSelfPlayDataSource


GENERATOR_VERSION="h1_mid_gold_multiseed_v1"
MID_RANGE=range(5,10)


def _write_json(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(data,indent=2),encoding="utf-8")


def _load_records(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _seed(base_seed,seed_index,game_index):
    return int(np.random.SeedSequence([base_seed,seed_index,game_index]).generate_state(1,dtype=np.uint32)[0])


def _state_at(history,ply):
    state=GomokuState.initial(6,4)
    for action in history[:ply]: state=state.play(action)
    return state


def original_result_record(state,history,result,seed,game_index,source_provenance=None):
    return make_record(
        state,history,result,seed,generator_version=GENERATOR_VERSION,ply=len(history),dedup_mode="d4",
        provenance_extra={**(source_provenance or {}),"state_source":"selfplay_mode_data_multiseed","game_index":game_index,"benchmark_exact_skip_policy":"disabled_always_try_verified_exact","label_authority":"original_verified_exact_solver"},
    )


def collect_task1(source,existing_records,dedup_reference_records,output_dir,n_min,k_seeds,games_per_seed,base_seed,budget_ms,node_cap):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); progress_path=output_dir/"task1_progress.json"; games_path=output_dir/"task1_games.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    completed_games=set(json.loads(games_path.read_text())) if games_path.exists() else set()
    source_provenance=source.provenance()
    # Avoid re-measuring any state already present in the E-3a.1 population,
    # not merely its exact-complete subset. This keeps the expansion genuinely
    # out-of-sample while the final artifact remains gold-only.
    seen={canonical_key(state_from_record(item)) for item in dedup_reference_records}
    for row in progress.values(): seen.add(canonical_key(_state_at(row["history"],row["ply"])))
    existing_mid=[item for item in existing_records if 5<=int(item["provenance"]["ply"])<=9]
    accepted_new=sum(row["status"]=="exact_complete" for row in progress.values())
    for seed_index in range(k_seeds):
        if len(existing_mid)+accepted_new>=n_min: break
        for game_index in range(games_per_seed):
            if len(existing_mid)+accepted_new>=n_min: break
            game_key=f"{seed_index}:{game_index}"
            if game_key in completed_games: continue
            seed=_seed(base_seed,seed_index,game_index); rng=random.Random(seed)
            # E-3a.2 only consumes ply 5..9, so generating beyond ply 9 adds
            # model inference cost without producing another candidate.
            _,history=source.trajectory(rng,6,4,9); routed=0
            for ply in MID_RANGE:
                if len(existing_mid)+accepted_new>=n_min or len(history)<ply: break
                state=_state_at(history,ply)
                if state.terminal(): continue
                key=canonical_key(state); state_id=state_identifier(state)
                if key in seen: continue
                seen.add(key)
                started=time.perf_counter(); result=route_ground_truth(state,GroundTruthBudget(node_cap,budget_ms),empty_bounds={})
                record=None
                if result.status=="exact_complete":
                    record=original_result_record(state,history[:ply],result,seed,game_index,source_provenance)
                    validation=validate_record(record)
                    if not validation.accepted or not validation.eligible: raise RuntimeError(f"invalid gold record {state_id}: {validation.errors}")
                    accepted_new+=1
                progress[state_id]={"state_id":state_id,"seed_index":seed_index,"seed":seed,"game_index":game_index,"ply":ply,"history":history[:ply],"empty_count":len(state.legal_actions()),"status":result.status,"value":result.value,"optimal_actions":None if result.optimal_actions is None else list(result.optimal_actions),"wall_ms":(time.perf_counter()-started)*1000,"record":record,"source_provenance":source_provenance}
                _write_json(progress_path,progress)
                routed+=1
                print(json.dumps({"stage":"task1","seed_index":seed_index,"game":game_index,"ply":ply,"status":result.status,"new_mid_complete":accepted_new,"mid_complete_total":len(existing_mid)+accepted_new,"target":n_min}),flush=True)
            completed_games.add(game_key); _write_json(games_path,sorted(completed_games))
            print(json.dumps({"stage":"game_complete","seed_index":seed_index,"game":game_index,"history_ply":len(history),"new_candidates":routed,"games_complete":len(completed_games)}),flush=True)
    return progress


def enhanced_propose_and_confirm(progress,existing_mid_count,output_dir,n_min,enhanced_budget_ms,confirm_budget_ms,node_cap):
    output_dir=Path(output_dir); enhanced_path=output_dir/"task2_progress.json"; enhanced=json.loads(enhanced_path.read_text()) if enhanced_path.exists() else {}
    accepted=sum(row["status"]=="exact_complete" for row in progress.values())
    for state_id,row in progress.items():
        if existing_mid_count+accepted>=n_min: break
        if row["status"]=="exact_complete" or state_id in enhanced: continue
        state=_state_at(row["history"],row["ply"]); proposal=solve_all_actions(state,time_cap_ms=enhanced_budget_ms,node_cap=node_cap,use_tt=True)
        item={"state_id":state_id,"enhanced_status":proposal.status,"enhanced_value":proposal.value,"enhanced_optimal_actions":None if proposal.optimal_actions is None else list(proposal.optimal_actions),"original_status":None,"match":None,"accepted":False}
        if proposal.status=="exact":
            confirmed=route_ground_truth(state,GroundTruthBudget(node_cap,confirm_budget_ms),empty_bounds={})
            item.update({"original_status":confirmed.status,"original_value":confirmed.value,"original_optimal_actions":None if confirmed.optimal_actions is None else list(confirmed.optimal_actions)})
            if confirmed.status=="exact_complete":
                item["match"]=(proposal.value==confirmed.value and proposal.optimal_actions==confirmed.optimal_actions)
                if item["match"]:
                    record=original_result_record(state,row["history"],confirmed,row["seed"],row["game_index"],row.get("source_provenance"))
                    if not validate_record(record).eligible: raise RuntimeError(f"invalid confirmed record {state_id}")
                    progress[state_id].update({"status":"exact_complete","value":confirmed.value,"optimal_actions":list(confirmed.optimal_actions),"record":record,"accepted_via":"enhanced_proposal_original_confirmation"})
                    item["accepted"]=True; accepted+=1
        enhanced[state_id]=item; _write_json(enhanced_path,enhanced); _write_json(output_dir/"task1_progress.json",progress)
        print(json.dumps({"stage":"task2","enhanced":proposal.status,"original":item["original_status"],"match":item["match"],"mid_complete_total":existing_mid_count+accepted}),flush=True)
    return progress,enhanced


def finalize(existing_records,progress,enhanced,output_dir,n_min,k_seeds):
    output_dir=Path(output_dir); candidate_new_records=[row["record"] for row in progress.values() if row.get("record")]
    # A second, generous run of the original verified solver is the final gold
    # gate. Candidates that timeout or disagree are logged and excluded.
    agreement=[]; new_records=[]
    for item in candidate_new_records:
        state=state_from_record(item); expected=item["solver"]
        exact=solve_actions(state,deadline_ms=30_000,node_budget=5_000_000)
        matched=exact.status=="exact" and exact.value==expected["value"] and list(exact.optimal_actions)==expected["optimal_actions"]
        agreement.append({"state_id":item["state_id"],"matched":matched,"expected_value":expected["value"],"expected_optimal_actions":expected["optimal_actions"],"actual_status":exact.status,"actual_value":exact.value,"actual_optimal_actions":list(exact.optimal_actions),"nodes":exact.nodes,"elapsed_ms":exact.elapsed_ms})
        if matched: new_records.append(item)
    _write_json(output_dir/"original_agreement.json",agreement)
    expanded=existing_records+new_records
    # D4 uniqueness is a hard gate across old and new gold.
    keys=[]
    for item in expanded:
        validation=validate_record(item)
        if not validation.accepted or item["solver"]["status"]!="exact_complete": raise RuntimeError("expanded gold contains invalid/non-complete record")
        keys.append(canonical_key(state_from_record(item)))
    if len(keys)!=len(set(keys)): raise RuntimeError("D4 duplicate in expanded gold")
    output_path=output_dir/"expanded_gold.jsonl"; output_path.write_text("".join(json.dumps(item,separators=(",",":"))+"\n" for item in expanded),encoding="utf-8")
    phase=Counter("mid" if 5<=int(item["provenance"]["ply"])<=9 else "late" for item in expanded); plys=[int(item["provenance"]["ply"]) for item in expanded]
    mismatch=sum(item.get("match") is False for item in enhanced.values()); task2_only=sum(item.get("accepted",False) for item in enhanced.values())
    branch="mid_vs_late_strong" if phase["mid"]>=30 else ("mid_suggestive" if phase["mid"]>=10 else "late_only")
    agreement_mismatches=sum(not row["matched"] for row in agreement)
    summary={"existing_gold":len(existing_records),"new_mid_candidates":len(candidate_new_records),"new_mid_complete":len(new_records),"mid_complete":phase["mid"],"late_complete":phase["late"],"total_gold":len(expanded),"gold_ply":{"min":min(plys),"median":float(np.median(plys)),"mean":float(np.mean(plys)),"max":max(plys),"counts":dict(sorted(Counter(plys).items()))},"new_gold_original_agreement":{"checks":len(agreement),"mismatches":agreement_mismatches,"retained":len(new_records),"rejected":agreement_mismatches,"budget_ms":30000,"node_cap":5000000},"task2":{"ran":bool(enhanced),"proposals":len(enhanced),"accepted_after_original_confirmation":task2_only,"mismatches":mismatch,"enhanced_only_labels":0},"k_seeds":k_seeds,"N_min":n_min,"branch":branch}
    _write_json(output_dir/"summary.json",summary); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--existing",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--n-min",type=int,default=30); parser.add_argument("--k-seeds",type=int,default=20); parser.add_argument("--games-per-seed",type=int,default=6); parser.add_argument("--base-seed",type=int,default=1000); parser.add_argument("--mcts-playouts",type=int,default=4); parser.add_argument("--device",default="cpu",help="Self-play inference device: cpu (recommended here), auto, cuda, or cuda:N"); parser.add_argument("--budget-ms",type=int,default=2000); parser.add_argument("--node-cap",type=int,default=1_000_000); parser.add_argument("--skip-task1",action="store_true",help="Resume directly from an existing task1_progress.json"); parser.add_argument("--enhanced-proposer",action="store_true"); parser.add_argument("--enhanced-budget-ms",type=int,default=2000); parser.add_argument("--confirm-budget-ms",type=int,default=8000); args=parser.parse_args()
    if not d4_roundtrip_self_check(): raise SystemExit("D4 round-trip failed")
    existing_population=[item for item in _load_records(args.existing) if item["state"]["board_size"]==6]
    existing=[item for item in existing_population if item["solver"]["status"]=="exact_complete"]
    if args.skip_task1:
        progress_path=args.output_dir/"task1_progress.json"
        if not progress_path.exists(): raise SystemExit("--skip-task1 requires task1_progress.json")
        progress=json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        source=ModelSelfPlayDataSource(args.checkpoint,"rgat",args.mcts_playouts,device=args.device)
        progress=collect_task1(source,existing,existing_population,args.output_dir,args.n_min,args.k_seeds,args.games_per_seed,args.base_seed,args.budget_ms,args.node_cap)
    enhanced={}
    current_mid=sum(5<=int(item["provenance"]["ply"])<=9 for item in existing)+sum(row["status"]=="exact_complete" for row in progress.values())
    if current_mid<args.n_min and args.enhanced_proposer: progress,enhanced=enhanced_propose_and_confirm(progress,sum(5<=int(item["provenance"]["ply"])<=9 for item in existing),args.output_dir,args.n_min,args.enhanced_budget_ms,args.confirm_budget_ms,args.node_cap)
    print(json.dumps(finalize(existing,progress,enhanced,args.output_dir,args.n_min,args.k_seeds),indent=2))


if __name__=="__main__": main()
