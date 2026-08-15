"""Four mandatory soundness gates for the enhanced offline exact solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.game import GomokuState
from azgomoku.offline_solver import solve_all_actions,solve_value,transform_action,transform_board
from azgomoku.solver import solve_actions


def h1_state(record):
    data=record["state"]
    return GomokuState(np.asarray(data["board"],dtype=np.int8),data["current_player"],data["last_move"],data["win_length"])


def transformed(state,symmetry):
    last=-1 if state.last_move<0 else transform_action(state.last_move,state.size,symmetry)
    return GomokuState(transform_board(state.board,symmetry).copy(),state.to_play,last,state.win_length)


def run(cache_path,audit_path,benchmark_path,output_path,vcf_crosscheck_path=None):
    cache=torch.load(cache_path,map_location="cpu",weights_only=False)
    audit=json.loads(Path(audit_path).read_text())["rows"]
    exhausted=[state for (state,_),(status,reason) in zip(cache["populations"][6],cache["measurements"][6]) if status=="unknown" and reason=="exhausted"]
    old_decided=[state for state,row in zip(exhausted,audit) if row["status"]=="exact"]
    h1=[json.loads(line) for line in Path(benchmark_path).read_text().splitlines() if line.strip()]

    agreement=[]; agreement_states=[]
    for state in old_decided:
        old=solve_actions(state); new=solve_all_actions(state,time_cap_ms=60_000,node_cap=10_000_000)
        ok=new.status==old.status=="exact" and new.value==old.value and new.optimal_actions==old.optimal_actions and new.action_values==old.action_values
        agreement.append(ok); agreement_states.append(state)
    for record in h1:
        state=h1_state(record); new=solve_all_actions(state,time_cap_ms=60_000,node_cap=10_000_000)
        expected_values={int(key):value for key,value in record["solver"]["action_values"].items()}
        ok=new.status=="exact" and new.value==record["solver"]["value"] and new.optimal_actions==tuple(record["solver"]["optimal_actions"]) and new.action_values==expected_values
        agreement.append(ok); agreement_states.append(state)
    gate1=all(agreement)

    symmetry_checks=[]
    for state in agreement_states[:5]:
        base=solve_all_actions(state,time_cap_ms=60_000,node_cap=10_000_000)
        for symmetry in range(8):
            result=solve_all_actions(transformed(state,symmetry),time_cap_ms=60_000,node_cap=10_000_000)
            mapped=tuple(sorted(transform_action(action,state.size,symmetry) for action in base.optimal_actions))
            symmetry_checks.append(result.status=="exact" and result.value==base.value and result.optimal_actions==mapped)
    gate2=all(symmetry_checks)

    vcf_states=[state for (state,_),(status,_) in zip(cache["populations"][6],cache["measurements"][6]) if status=="exact_partial"]
    vcf_checks=[]
    if vcf_crosscheck_path is not None:
        crosscheck=json.loads(Path(vcf_crosscheck_path).read_text())
        for state in vcf_states:
            row=crosscheck.get(state_identifier(state))
            vcf_checks.append(row is not None and row["status"]=="exact" and row["value"]==1)
    else:
        for state in vcf_states:
            result=solve_value(state,time_cap_ms=60_000,node_cap=10_000_000,use_tt=True)
            vcf_checks.append(result.status=="exact" and result.value==1)
    gate3=all(vcf_checks)

    tt_checks=[]
    for state in agreement_states[:10]:
        with_tt=solve_all_actions(state,time_cap_ms=60_000,node_cap=10_000_000,use_tt=True)
        without_tt=solve_all_actions(state,time_cap_ms=60_000,node_cap=10_000_000,use_tt=False)
        tt_checks.append(with_tt.status==without_tt.status=="exact" and with_tt.value==without_tt.value and with_tt.optimal_actions==without_tt.optimal_actions and with_tt.action_values==without_tt.action_values)
    gate4=all(tt_checks)
    report={"agreement_old_and_h1":{"passed":gate1,"checks":len(agreement),"failures":len(agreement)-sum(agreement)},"d4_self_consistency":{"passed":gate2,"checks":len(symmetry_checks),"failures":len(symmetry_checks)-sum(symmetry_checks)},"vcf_consistency":{"passed":gate3,"checks":len(vcf_checks),"failures":len(vcf_checks)-sum(vcf_checks)},"tt_on_off":{"passed":gate4,"checks":len(tt_checks),"failures":len(tt_checks)-sum(tt_checks)},"all_passed":gate1 and gate2 and gate3 and gate4}
    Path(output_path).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2)); return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--audit",type=Path,required=True); parser.add_argument("--benchmark",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--vcf-crosscheck",type=Path); args=parser.parse_args()
    run(args.cache,args.audit,args.benchmark,args.output,args.vcf_crosscheck)


if __name__=="__main__": main()
