import csv,json,platform,time
from pathlib import Path
import numpy as np
import torch
from .game import GomokuState
from .mcts import search
from .replay import ReplayBuffer
from .evaluation import evaluate_vs_random,inspect_tactics

def self_play(model,cfg):
    samples=[]; lengths=[]; start=time.perf_counter()
    for _ in range(cfg.self_play_games):
        state=GomokuState.initial(cfg.board_size,cfg.win_length); game=[]
        while not state.terminal():
            pi=search(model,state,cfg.mcts_playouts,cfg.c_puct,cfg.temperature); game.append((state.features(),pi,state.to_play))
            state=state.play(np.random.choice(cfg.board_size**2,p=pi))
        winner=state.winner(); lengths.append(len(game)); samples.extend((x,pi,float(int(winner==p)-int(winner==-p))) for x,pi,p in game)
    return samples,{"game_length":float(np.mean(lengths)),"self_play_time":time.perf_counter()-start}

def train(model,replay,cfg):
    opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); rows=[]
    for _ in range(cfg.train_epochs):
        batch=replay.sample(cfg.batch_size); x=torch.tensor(np.stack([b[0] for b in batch])); pi=torch.tensor(np.stack([b[1] for b in batch])); z=torch.tensor([b[2] for b in batch])
        logits,v=model(x); logp=torch.log_softmax(logits,1); pl=-(pi*logp).sum(1).mean(); vl=torch.mean((z-v)**2); loss=pl+vl
        opt.zero_grad(); loss.backward(); opt.step(); entropy=-(torch.softmax(logits,1)*logp).sum(1).mean()
        rows.append({"policy_loss":pl.item(),"value_loss":vl.item(),"total_loss":loss.item(),"policy_entropy":entropy.item()})
    return rows

def latency_ms(model,size,runs=20):
    x=torch.zeros(1,6,size,size); model.eval()
    with torch.no_grad():
        for _ in range(3): model(x)
        t=time.perf_counter()
        for _ in range(runs): model(x)
    return 1000*(time.perf_counter()-t)/runs

def run_experiment(name,model,cfg,outdir):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    replay=ReplayBuffer(cfg.replay_capacity); samples,sp=self_play(model,cfg); replay.extend(samples); rows=train(model,replay,cfg); lat=latency_ms(model,cfg.board_size); evaluation=evaluate_vs_random(model,cfg)
    for row in rows: row.update(sp); row["inference_latency_ms"]=lat
    (out/"config.json").write_text(json.dumps(cfg.dict(),indent=2)); torch.save(model.state_dict(),out/"model.pt")
    with (out/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    runtime={"python":platform.python_version(),"pytorch":torch.__version__,"device":"cpu","inference_latency_ms":lat}; (out/"runtime.json").write_text(json.dumps(runtime,indent=2))
    inspect_tactics(model,out/"tactical_inspection.json")
    (out/"summary.md").write_text(f"# {name} summary\n\nProfile completed with {len(samples)} positions. Batch-1 latency: {lat:.3f} ms. Evaluation: {evaluation}.\n\nTactical examples are inspection-only. Attention weights, where present, are not causal explanations.\n")
    return rows
