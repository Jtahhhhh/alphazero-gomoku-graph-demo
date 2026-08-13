"""Controlled R-GCN/R-GAT H3 pilot trainer with immutable checkpoint/resume."""

import argparse,csv,json,time
from pathlib import Path
import torch

from azgomoku.config import Config
from azgomoku.h3_checkpoint import load_bundle,make_bundle,save_bundle
from azgomoku.replay import ReplayBuffer
from azgomoku.reproducibility import seed_everything
from azgomoku.training import self_play,train
from models.rgcn import RGCN
from models.rgat import RGAT


MODELS={"rgcn":RGCN,"rgat":RGAT}


def load_config(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def as_training_config(data):
    return Config(board_size=data["board_size"],win_length=data["win_length"],mcts_playouts=data["mcts_playouts"],self_play_games=data["selfplay_games_per_iter"],replay_capacity=data["replay_capacity"],batch_size=data["batch_size"],learning_rate=data["learning_rate"],weight_decay=data["weight_decay"],hidden_dim=data["hidden_dim"],attention_heads=data["attention_heads"],train_epochs=data["train_updates_per_iter"],training_iterations=data["iterations"],c_puct=data["c_puct"],temperature=data["temperature"],opening_temperature_moves=data.get("opening_temperature_moves",10),late_temperature=data.get("late_temperature",0.0),dirichlet_alpha=data.get("dirichlet_alpha",0.3),dirichlet_fraction=data.get("dirichlet_fraction",0.25),symmetry_augmentation=data.get("symmetry_augmentation",True),seed=data["seed"])


def append_rows(path,rows):
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));
        if not exists: writer.writeheader()
        writer.writerows(rows)


def run(config_path,output,resume=None,max_iterations=None):
    data=load_config(config_path); model_type=data["model_type"]
    if model_type not in MODELS: raise ValueError("H3 pilot supports only rgcn and rgat")
    seed_everything(data["seed"]); cfg=as_training_config(data)
    model=MODELS[model_type](board_size=cfg.board_size,hidden_dim=cfg.hidden_dim,attention_heads=cfg.attention_heads)
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); replay=ReplayBuffer(cfg.replay_capacity)
    output=Path(output); output.mkdir(parents=True,exist_ok=True); (output/"config.json").write_text(json.dumps(data,indent=2),encoding="utf-8")
    state={"iteration":0,"selfplay_games_seen":0,"optimizer_updates":0,"replay_size":0,"positions_seen":0}; start=1
    if resume:
        bundle=load_bundle(resume,model,optimizer,replay,model_type); state=dict(bundle["training_state"]); start=state["iteration"]+1
    elif not (output/"checkpoints"/"iter_000.pt").exists(): save_bundle(output,make_bundle(model_type,model,optimizer,replay,data,state))
    stop=min(data["iterations"],max_iterations or data["iterations"]); log_path=output/"training_log.csv"
    for iteration in range(start,stop+1):
        iteration_start=time.perf_counter(); selfplay_start=time.perf_counter(); samples,sp=self_play(model,cfg); selfplay_s=time.perf_counter()-selfplay_start
        replay.extend(samples); train_start=time.perf_counter(); rows=train(model,replay,cfg,optimizer); train_s=time.perf_counter()-train_start
        state={"iteration":iteration,"selfplay_games_seen":state["selfplay_games_seen"]+cfg.self_play_games,"optimizer_updates":state["optimizer_updates"]+cfg.train_epochs,"replay_size":len(replay),"positions_seen":state["positions_seen"]+len(samples)}
        for update,row in enumerate(rows,1): row.update({**state,"update_in_iteration":update,"selfplay_seconds":selfplay_s,"training_seconds":train_s,"iteration_seconds":time.perf_counter()-iteration_start,**{k:v for k,v in sp.items() if k!="self_play_time"}})
        append_rows(log_path,rows); print(json.dumps({"event":"h3_iteration","model":model_type,**state,"selfplay_seconds":round(selfplay_s,3),"training_seconds":round(train_s,3)}),flush=True)
        if iteration%data["checkpoint_every"]==0 or iteration==stop: save_bundle(output,make_bundle(model_type,model,optimizer,replay,data,state))
    torch.save(model.state_dict(),output/"model.pt"); return state


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--resume",type=Path); parser.add_argument("--max-iterations",type=int); args=parser.parse_args()
    run(args.config,args.output,args.resume,args.max_iterations)


if __name__=="__main__": main()
