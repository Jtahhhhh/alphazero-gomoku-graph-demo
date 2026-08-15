"""Join certified H1 ground truth to network evidence and MCTS-v2 metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.game import GomokuState
from azgomoku.mcts import search
from azgomoku.metrics.semantic_alignment import (
    aggregate_proofs,
    average_precision,
    baselines,
    critical_ids,
    entropy,
    score_alignment,
)
from models.rgat import RGAT
from models.rgcn import RGCN


def load_records(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def to_state(record):
    data=record["state"]
    return GomokuState(np.asarray(data["board"],dtype=np.int8),int(data["current_player"]),int(data["last_move"]),int(data["win_length"]))


def evaluate_record(record,model,model_name,checkpoint,playouts):
    state=to_state(record); optimal=set(map(int,record["solver"]["optimal_actions"])); legal=list(map(int,state.legal_actions()))
    x=torch.from_numpy(state.features()).unsqueeze(0)
    with torch.no_grad(): logits,value=model(x,return_evidence=False)
    mask=torch.full_like(logits,-torch.inf); mask[0,legal]=logits[0,legal]; policy=torch.softmax(mask,dim=-1)[0].cpu().numpy(); policy_action=int(policy.argmax())
    evaluation_action=policy_action if policy_action in optimal else min(optimal)
    evidence=collect_model_evidence(state,model,evaluation_action); edges=evidence["graph_evidence"]["edges"]
    proofs=[proof for proof in record["valid_proofs"] if proof["action"]==evaluation_action]
    if not proofs: proofs=[proof for proof in record["valid_proofs"] if proof["action"] in optimal]
    scores=[edge["attention"] for edge in edges]; graph=aggregate_proofs(edges,scores,proofs); random_base,structural_base=baselines(edges,proofs,record["state_id"])
    per_head={}
    if model_name=="rgat" and edges:
        for head in range(len(edges[0]["head_attention"])): per_head[str(head)]=aggregate_proofs(edges,[edge["head_attention"][head] for edge in edges],proofs)["mean"]
    pi,root=search(model,state,playouts=playouts,temperature=1.0,return_root=True); selected=int(pi.argmax())
    concept=proofs[0]["concepts"][0] if proofs else "none"; relation=proofs[0]["critical_relations"][0] if proofs else "none"
    return {
        "run_id":"h1_mvp_seed7","model_type":model_name,"checkpoint":str(checkpoint),"state_id":record["state_id"],
        "solver_value":record["solver"]["value"],"num_optimal_actions":len(optimal),"num_valid_proofs":len(record["valid_proofs"]),"concept":concept,"relation":relation,
        "policy_top1_correct":int(policy_action in optimal),"policy_optimal_mass":float(sum(policy[a] for a in optimal)),"policy_entropy":entropy(policy[legal]),
        "value_prediction":float(value.item()),"value_error":abs(float(value.item())-record["solver"]["value"]),
        "graph_critical_mass":graph["mean"].get("mass",0),"graph_best_valid_proof_mass":graph["best"].get("mass",0),"graph_precision_at_k":graph["mean"].get("precision_at_k",0),"graph_recall_at_k":graph["mean"].get("recall_at_k",0),"graph_auprc":graph["mean"].get("auprc",0),
        "random_critical_mass":random_base.get("mass",0),"structural_critical_mass":structural_base.get("mass",0),"per_head_metrics_json":json.dumps(per_head,separators=(",",":")),
        "mcts_selected_optimal":int(selected in optimal),"mcts_optimal_mass":float(sum(pi[a] for a in optimal)),"mcts_entropy":entropy(pi[legal]),"search_gain_optimal_mass":float(sum(pi[a] for a in optimal)-sum(policy[a] for a in optimal)),
        "mcts_value_convention_version":2,"q_perspective":"player_who_selects_action_at_parent",
    },{"ground_truth":record,"evaluation_action":evaluation_action,"network":evidence["network"],"graph_evidence":evidence["graph_evidence"],"mcts":{"selected_action":selected,"pi":[float(v) for v in pi],"candidates":[{"action":a,"P":float(c.prior),"N":int(c.n),"Q":float(c.q),"pi":float(pi[a])} for a,c in sorted(root.children.items())],"mcts_value_convention_version":2,"q_perspective":"player_who_selects_action_at_parent"}}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--benchmark",type=Path,default=Path("diagnostic/h1_tactical.jsonl")); parser.add_argument("--output",type=Path,default=Path("results/h1/h1_metrics.csv")); parser.add_argument("--playouts",type=int,default=20); args=parser.parse_args()
    records=load_records(args.benchmark); configs=[("rgat",RGAT(),Path("results/h3_pilot/rgat/seed_7/model.pt")),("rgcn",RGCN(),Path("results/h3_pilot/rgcn/seed_7/model.pt"))]; rows=[]; example=None
    for name,model,checkpoint in configs:
        model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)); model.eval()
        for record in records:
            row,evidence=evaluate_record(record,model,name,checkpoint,args.playouts); rows.append(row)
            if example is None and name=="rgat" and any("diagonal" in relation for proof in record["valid_proofs"] for relation in proof["critical_relations"] if "immediate_win" in proof["concepts"]): example={"model_type":name,**evidence,"metrics":row}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output.parent/"h1_example_evidence.json").write_text(json.dumps(example,indent=2),encoding="utf-8")
    summary={"states":len(records),"rows":len(rows),"models":sorted({row["model_type"] for row in rows}),"mean":{key:float(np.mean([float(row[key]) for row in rows])) for key in ("policy_optimal_mass","value_error","graph_critical_mass","random_critical_mass","structural_critical_mass","mcts_optimal_mass","search_gain_optimal_mass")}}
    (args.output.parent/"h1_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
