"""Resumable enhanced-exact cross-check over replay-verified VCF positives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.offline_solver import solve_value
from azgomoku.vcf import solve_vcf


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--budget-ms",type=int,default=60_000); parser.add_argument("--retry-unknown",action="store_true"); parser.add_argument("--vcf-ordering-hint",action="store_true"); args=parser.parse_args()
    cache=torch.load(args.cache,map_location="cpu",weights_only=False)
    states=[state for (state,_),(status,_) in zip(cache["populations"][6],cache["measurements"][6]) if status=="exact_partial"]
    rows=json.loads(args.output.read_text()) if args.output.exists() else {}
    for index,state in enumerate(states):
        key=state_identifier(state)
        if key not in rows or (args.retry_unknown and rows[key]["status"] != "exact"):
            preferred_action=None
            if args.vcf_ordering_hint:
                vcf=solve_vcf(state,node_cap=10_000,time_cap_ms=250)
                if vcf.status!="exact_partial" or not vcf.optimal_actions:
                    raise RuntimeError(f"VCF ordering hint unavailable for {key}")
                preferred_action=vcf.optimal_actions[0]
            result=solve_value(state,time_cap_ms=args.budget_ms,node_cap=10_000_000,use_tt=True,preferred_action=preferred_action)
            rows[key]={"status":result.status,"value":result.value,"nodes":result.nodes,"elapsed_ms":result.elapsed_ms}
            args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(rows,indent=2),encoding="utf-8")
        print(json.dumps({"done":index+1,"total":len(states),"state_id":key,**rows[key]}),flush=True)
    state_keys={state_identifier(state) for state in states}
    values=[rows[key] for key in state_keys if key in rows]
    summary={"checks":len(states),"recorded":len(values),"exact_plus_one":sum(row["status"]=="exact" and row["value"]==1 for row in values),"exact_contradictions":sum(row["status"]=="exact" and row["value"]!=1 for row in values),"unresolved_or_missing":len(states)-sum(row["status"]=="exact" for row in values)}
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
