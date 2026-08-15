"""D2c-v2 Track 1: calibrate an exact-decidable empty-count bound."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.offline_solver import solve_value
from azgomoku.solver import solve_actions


def render_svg(rows):
    width,height=920,500; left,bottom,top,right=70,440,45,30
    max_empty=max(row["empty_count"] for row in rows); min_empty=min(row["empty_count"] for row in rows)
    max_ms=max(1,max(row["elapsed_ms"] for row in rows))
    def x(empty): return left+(empty-min_empty)/max(1,max_empty-min_empty)*(width-left-right)
    def y(ms): return bottom-min(ms,max_ms)/max_ms*(bottom-top)
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f8fafc"/><text x="{width/2}" y="25" text-anchor="middle" font-family="Arial" font-size="18" font-weight="bold">Exact solve time vs empty cells</text><line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#334155"/><line x1="{left}" y1="{bottom}" x2="{width-right}" y2="{bottom}" stroke="#334155"/>']
    for row in rows:
        color="#16a34a" if row["status"]=="exact" else "#dc2626"
        parts.append(f'<circle cx="{x(row["empty_count"]):.2f}" cy="{y(row["elapsed_ms"]):.2f}" r="4" fill="{color}" opacity=".75"/>')
    for empty in range(min_empty,max_empty+1): parts.append(f'<text x="{x(empty):.2f}" y="{bottom+20}" text-anchor="middle" font-family="Arial" font-size="10">{empty}</text>')
    parts.append(f'<text x="{width/2}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="12">empty cells</text><text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="12">elapsed ms (red = timeout)</text></svg>')
    return "".join(parts)


def run(cache_path,output_dir,budget_ms=60_000,node_cap=10_000_000,solver_mode="current"):
    cache=torch.load(cache_path,map_location="cpu",weights_only=False)
    u6=[]
    for (state,ply),(status,reason) in zip(cache["populations"][6],cache["measurements"][6]):
        if status=="unknown": u6.append((state,ply,reason))
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    progress_path=output_dir/"track1_progress.json"
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    by_empty=defaultdict(list)
    for state,ply,reason in u6: by_empty[len(state.legal_actions())].append((state,ply,reason))
    stop_after_failure=False
    for empty in sorted(by_empty):
        if stop_after_failure: break
        for index,(state,ply,reason) in enumerate(by_empty[empty]):
            key=state_identifier(state)
            if key in progress: continue
            result=(solve_actions(state,deadline_ms=budget_ms,node_budget=node_cap) if solver_mode=="current" else solve_value(state,time_cap_ms=budget_ms,node_cap=node_cap,use_tt=True))
            progress[key]={"state_id":key,"empty_count":empty,"ply":ply,"vcf_unknown_reason":reason,"status":result.status,"value":result.value,"nodes":result.nodes,"elapsed_ms":result.elapsed_ms}
            progress_path.write_text(json.dumps(progress,indent=2),encoding="utf-8")
            print(json.dumps({"empty":empty,"done":index+1,"states":len(by_empty[empty]),"status":result.status,"elapsed_ms":result.elapsed_ms}),flush=True)
        bucket=[row for row in progress.values() if row["empty_count"]==empty]
        completion=sum(row["status"]=="exact" for row in bucket)/len(bucket)
        print(json.dumps({"empty_bucket":empty,"completion":completion,"states":len(bucket)}),flush=True)
        if completion<.95: stop_after_failure=True
    rows=sorted(progress.values(),key=lambda row:(row["empty_count"],row["state_id"]))
    buckets=[]
    for empty in sorted({row["empty_count"] for row in rows}):
        group=[row for row in rows if row["empty_count"]==empty]
        buckets.append({"empty_count":empty,"states":len(group),"exact":sum(row["status"]=="exact" for row in group),"completion":sum(row["status"]=="exact" for row in group)/len(group),"median_elapsed_ms":sorted(row["elapsed_ms"] for row in group)[len(group)//2]})
    passing=[row["empty_count"] for row in buckets if row["completion"]>=.95]
    e_star=max(passing) if passing else None
    bounded=[row for row in rows if e_star is not None and row["empty_count"]<=e_star]
    unresolved=[row for row in bounded if row["vcf_unknown_reason"]=="exhausted" and row["status"]!="exact"]
    counts={"A":0,"B":0,"C":0}
    for row in bounded:
        if row["vcf_unknown_reason"]=="budget": counts["C"]+=1
        elif row["status"]=="exact" and row["value"]==1: counts["A"]+=1
        elif row["status"]=="exact": counts["B"]+=1
    denominator=len(bounded); p_a=counts["A"]/denominator if denominator and not unresolved else None
    summary={"solver_mode":solver_mode,"budget_ms":budget_ms,"node_cap":node_cap,"U6":len(u6),"buckets":buckets,"E_star":e_star,"U6_dec":denominator,"A":counts["A"],"B":counts["B"],"C":counts["C"],"unresolved_in_bound":len(unresolved),"P_A_dec":p_a}
    (output_dir/"track1_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    with (output_dir/"solve_time_vs_empty.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output_dir/"solve_time_vs_empty.svg").write_text(render_svg(rows),encoding="utf-8")
    print(json.dumps(summary,indent=2)); return summary


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--cache",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--budget-ms",type=int,default=60_000); parser.add_argument("--node-cap",type=int,default=10_000_000); parser.add_argument("--solver-mode",choices=("current","enhanced"),default="current"); args=parser.parse_args()
    run(args.cache,args.output_dir,args.budget_ms,args.node_cap,args.solver_mode)


if __name__=="__main__": main()
