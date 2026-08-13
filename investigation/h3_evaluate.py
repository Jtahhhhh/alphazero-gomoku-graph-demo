"""Evaluate immutable H3 checkpoints on the frozen H1 benchmark."""

import argparse,csv,json,math,time
from pathlib import Path
import numpy as np
import torch

from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.game import GomokuState
from azgomoku.graph import cell_edge_records
from azgomoku.h3_checkpoint import model_from_bundle
from azgomoku.mcts import search
from investigation.evaluate_h1 import aggregate_proofs,baselines,entropy
from models.rgcn import RGCN
from models.rgat import RGAT


MODELS={"rgcn":RGCN,"rgat":RGAT}


def load_records(path): return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
def to_state(record):
    state=record["state"]; return GomokuState(np.asarray(state["board"],dtype=np.int8),state["current_player"],state["last_move"],state["win_length"])
def fixed_targets(record): return {"state_id":record["state_id"],"solver_value":record["solver"]["value"],"optimal_actions":tuple(sorted(record["solver"]["optimal_actions"])),"valid_proofs":json.dumps(record["valid_proofs"],sort_keys=True)}


def evaluate_state(record,model,model_type,iteration,metadata,mcts_playouts=None):
    target=fixed_targets(record); state=to_state(record); legal=list(map(int,state.legal_actions())); optimal=set(target["optimal_actions"])
    x=torch.from_numpy(state.features()).unsqueeze(0)
    with torch.no_grad(): logits,value=model(x,return_evidence=False)
    masked=torch.full_like(logits,-torch.inf); masked[0,legal]=logits[0,legal]; policy=torch.softmax(masked,dim=-1)[0].cpu().numpy(); action=int(policy.argmax())
    evidence=collect_model_evidence(state,model,action)
    if model_type=="rgat": edges=evidence["graph_evidence"]["edges"]; scores=[edge["attention"] for edge in edges]
    else:
        records=cell_edge_records(state.size); edges=[{"edge_id":e["edge_id"],"relation":e["relation"],"source":{"action":e["source"]},"target":{"action":e["target"]}} for e in records]
        indegree={}
        for edge in edges: key=(edge["relation"],edge["target"]["action"]); indegree[key]=indegree.get(key,0)+1
        scores=[1/indegree[(edge["relation"],edge["target"]["action"])] for edge in edges]
    alignment=aggregate_proofs(edges,scores,record["valid_proofs"])["mean"]; random_base,structural_base=baselines(edges,record["valid_proofs"],record["state_id"])
    row={"run_id":metadata["config"]["run_id"],"model_type":model_type,"seed":metadata["seed"],"iteration":iteration,"state_id":target["state_id"],"policy_top1_correct":int(action in optimal),"policy_optimal_mass":float(sum(policy[a] for a in optimal)),"policy_entropy":entropy(policy[legal]),"value_prediction":float(value),"solver_value":target["solver_value"],"value_error":abs(float(value)-target["solver_value"]),"graph_critical_mass":alignment.get("mass",0),"graph_precision_at_k":alignment.get("precision_at_k",0),"graph_recall_at_k":alignment.get("recall_at_k",0),"graph_auprc":alignment.get("auprc",0),"random_critical_mass":random_base.get("mass",0),"structural_critical_mass":structural_base.get("mass",0),"alignment_minus_random":alignment.get("mass",0)-random_base.get("mass",0),"alignment_minus_structural":alignment.get("mass",0)-structural_base.get("mass",0),"mcts_optimal_mass":None,"mcts_selected_optimal":None,"mcts_entropy":None,"search_gain":None,"optimizer_updates":metadata["training_state"]["optimizer_updates"],"selfplay_games_seen":metadata["training_state"]["selfplay_games_seen"]}
    if mcts_playouts:
        pi,_=search(model,state,playouts=mcts_playouts,temperature=1.0,return_root=True); selected=int(pi.argmax()); row.update({"mcts_optimal_mass":float(sum(pi[a] for a in optimal)),"mcts_selected_optimal":int(selected in optimal),"mcts_entropy":entropy(pi[legal]),"search_gain":float(sum(pi[a] for a in optimal)-sum(policy[a] for a in optimal))})
    return row


def evaluate_run(run_dir,benchmark,output):
    run_dir=Path(run_dir); records=load_records(benchmark); manifest=json.loads((run_dir/"checkpoints"/"manifest.json").read_text(encoding="utf-8")); rows=[]; timings=[]
    for entry in manifest["checkpoints"]:
        path=run_dir/"checkpoints"/entry["path"]; model,bundle=model_from_bundle(path,MODELS); iteration=entry["iteration"]; start=time.perf_counter(); mcts=50 if iteration in bundle["config"]["mcts_eval_iterations"] else None
        for record in records: rows.append(evaluate_state(record,model,bundle["model_type"],iteration,bundle,mcts))
        timings.append({"iteration":iteration,"seconds":time.perf_counter()-start,"mcts_enabled":bool(mcts)})
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as handle: writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output.parent/(output.stem+"_runtime.json")).write_text(json.dumps(timings,indent=2),encoding="utf-8"); return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--benchmark",type=Path,default=Path("diagnostic/h1_tactical.jsonl")); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); evaluate_run(args.run_dir,args.benchmark,args.output)


if __name__=="__main__": main()
