"""Run E-3a integration gates over legacy exact H1 and generated v2 records."""

from __future__ import annotations

import argparse,json
from pathlib import Path

from azgomoku.ground_truth import GroundTruthBudget,RouterStats,route_ground_truth
from azgomoku.h1_schema import state_from_record,validate_record


def run(legacy_path,candidates_path,output_path,limit=None):
    legacy=[json.loads(line) for line in Path(legacy_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None: legacy=legacy[:limit]
    stats=RouterStats(); mismatches=[]
    for index,item in enumerate(legacy):
        state=state_from_record(item); expected=item["solver"]
        result=route_ground_truth(state,GroundTruthBudget(1_000_000,2_000),stats=stats)
        if result.status!="exact_complete" or result.value!=expected["value"] or list(result.optimal_actions)!=expected["optimal_actions"]:
            mismatches.append({"index":index,"state_id":item["state_id"],"status":result.status})
    candidates=[json.loads(line) for line in Path(candidates_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    invalid=[]; partial=unknown=eligible=0
    for index,item in enumerate(candidates):
        validation=validate_record(item)
        if not validation.accepted: invalid.append({"index":index,"errors":list(validation.errors)})
        else:
            eligible+=int(validation.eligible); partial+=item["solver"]["status"]=="exact_partial"; unknown+=item["solver"]["status"]=="unknown"
    report={
        "legacy_exact_agreement":{"checks":len(legacy),"mismatches":len(mismatches),"details":mismatches,"router":stats.dict()},
        "v2_candidates":{"records":len(candidates),"invalid":len(invalid),"partial_replay_passed":partial,"unknown_outside_denominator":unknown,"eligible":eligible,"details":invalid},
        "passed":not mismatches and not invalid,
    }
    Path(output_path).parent.mkdir(parents=True,exist_ok=True); Path(output_path).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2)); return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--legacy",type=Path,default=Path("diagnostic/h1_tactical.jsonl")); parser.add_argument("--candidates",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--limit",type=int); args=parser.parse_args()
    if not run(args.legacy,args.candidates,args.output,args.limit)["passed"]: raise SystemExit(1)


if __name__=="__main__": main()
