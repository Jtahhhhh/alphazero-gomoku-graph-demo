import csv,json,platform,random,time
from pathlib import Path
import numpy as np
import torch
from .game import GomokuState
from .mcts import search
from .replay import ReplayBuffer
from .evaluation import evaluate_vs_random,inspect_tactics

def symmetry_augment(features,policy):
    """Return the eight D4 transforms with spatial features/policy aligned."""
    size=features.shape[-1]; policy_board=policy.reshape(size,size); augmented=[]
    for turns in range(4):
        x=np.rot90(features,turns,axes=(-2,-1)).copy(); pi=np.rot90(policy_board,turns).copy()
        augmented.append((x,pi.reshape(-1)))
        augmented.append((np.flip(x,axis=-1).copy(),np.fliplr(pi).copy().reshape(-1)))
    return augmented

def self_play(model,cfg,iteration=None):
    samples=[]; lengths=[]; opening_actions=[]; opening_corner_mass=[]; opening_edge_mass=[]; start=time.perf_counter()
    for game_index in range(cfg.self_play_games):
        if iteration is not None:
            print(f"[SELFPLAY] iter={iteration} game={game_index + 1}/{cfg.self_play_games} started",flush=True)
        state=GomokuState.initial(cfg.board_size,cfg.win_length); game=[]; ply=0
        while not state.terminal():
            temperature=cfg.temperature if ply<cfg.opening_temperature_moves else cfg.late_temperature
            pi=search(model,state,cfg.mcts_playouts,cfg.c_puct,temperature,dirichlet_alpha=cfg.dirichlet_alpha,dirichlet_fraction=cfg.dirichlet_fraction)
            if ply==0:
                size=cfg.board_size; corners=(0,size-1,size*(size-1),size*size-1); edges=[a for a in range(size*size) if a//size in (0,size-1) or a%size in (0,size-1)]
                opening_corner_mass.append(float(sum(pi[a] for a in corners))); opening_edge_mass.append(float(sum(pi[a] for a in edges)))
            action=int(np.random.choice(cfg.board_size**2,p=pi));
            if ply==0: opening_actions.append(action)
            game.append((state.features(),pi,state.to_play)); state=state.play(action); ply+=1
        winner=state.winner(); lengths.append(len(game))
        for x,pi,p in game:
            value=float(int(winner==p)-int(winner==-p))
            transformed=symmetry_augment(x,pi) if cfg.symmetry_augmentation else ((x,pi),)
            samples.extend((tx,tpi,value) for tx,tpi in transformed)
        if iteration is not None:
            elapsed=time.perf_counter()-start
            completed=game_index+1
            avg=elapsed/completed
            eta=avg*(cfg.self_play_games-completed)
            print(f"[SELFPLAY] iter={iteration} game={completed}/{cfg.self_play_games} done moves={len(game)} "
                  f"samples={len(samples)} games/sec={completed/elapsed:.3f} avg={avg:.1f}s/game ETA={eta:.1f}s",flush=True)
    opening_counts=np.bincount(opening_actions,minlength=cfg.board_size**2) if opening_actions else np.zeros(cfg.board_size**2)
    opening_probs=opening_counts/opening_counts.sum() if opening_counts.sum() else opening_counts
    opening_entropy=float(-(opening_probs[opening_probs>0]*np.log(opening_probs[opening_probs>0])).sum())
    return samples,{"game_length":float(np.mean(lengths)),"self_play_time":time.perf_counter()-start,"opening_unique_actions":int(np.count_nonzero(opening_counts)),"opening_entropy":opening_entropy,"opening_corner_mass":float(np.mean(opening_corner_mass)),"opening_edge_mass":float(np.mean(opening_edge_mass))}

def train(model,replay,cfg,opt=None,iteration=None,device=None):
    opt=opt or torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); rows=[]
    device=device or next(model.parameters()).device
    for update in range(cfg.train_epochs):
        batch=replay.sample(cfg.batch_size); x=torch.as_tensor(np.stack([b[0] for b in batch]),device=device); pi=torch.as_tensor(np.stack([b[1] for b in batch]),device=device); z=torch.as_tensor([b[2] for b in batch],device=device)
        logits,v=model(x); logp=torch.log_softmax(logits,1); pl=-(pi*logp).sum(1).mean(); vl=torch.mean((z-v)**2); loss=pl+vl
        opt.zero_grad(); loss.backward(); opt.step(); entropy=-(torch.softmax(logits,1)*logp).sum(1).mean()
        rows.append({"policy_loss":pl.item(),"value_loss":vl.item(),"total_loss":loss.item(),"policy_entropy":entropy.item()})
        if iteration is not None:
            print(f"[TRAIN] iter={iteration} update={update + 1}/{cfg.train_epochs} loss={loss.item():.6f} policy={pl.item():.6f} value={vl.item():.6f}",flush=True)
    return rows

def latency_ms(model,size,runs=20):
    x=torch.zeros(1,6,size,size); model.eval()
    with torch.no_grad():
        for _ in range(3): model(x)
        t=time.perf_counter()
        for _ in range(runs): model(x)
    return 1000*(time.perf_counter()-t)/runs

def run_experiment(name,model,cfg,outdir):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed); out=Path(outdir); out.mkdir(parents=True,exist_ok=True)
    replay=ReplayBuffer(cfg.replay_capacity); opt=torch.optim.Adam(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); rows=[]; total_samples=0
    print(json.dumps({"event":"run_start","model":name,"training_iterations":cfg.training_iterations,"self_play_games_per_iteration":cfg.self_play_games,"mcts_playouts":cfg.mcts_playouts,"train_epochs_per_iteration":cfg.train_epochs,"seed":cfg.seed}),flush=True)
    for iteration in range(1,cfg.training_iterations+1):
        print(json.dumps({"event":"self_play_start","model":name,"iteration":iteration,"iterations_total":cfg.training_iterations,"games":cfg.self_play_games}),flush=True)
        samples,sp=self_play(model,cfg); replay.extend(samples); total_samples+=len(samples); iteration_rows=train(model,replay,cfg,opt)
        for epoch,row in enumerate(iteration_rows,1):
            row.update(sp); row.update({"training_iteration":iteration,"epoch":epoch,"self_play_games_total":iteration*cfg.self_play_games,"positions_total":total_samples,"replay_size":len(replay)})
        rows.extend(iteration_rows)
        last=iteration_rows[-1]
        print(json.dumps({"event":"iteration_complete","model":name,"iteration":iteration,"iterations_total":cfg.training_iterations,"positions_generated":len(samples),"positions_total":total_samples,"replay_size":len(replay),"self_play_seconds":round(sp["self_play_time"],3),"mean_game_length":round(sp["game_length"],3),"policy_loss":round(last["policy_loss"],6),"value_loss":round(last["value_loss"],6),"total_loss":round(last["total_loss"],6),"policy_entropy":round(last["policy_entropy"],6)}),flush=True)
    print(json.dumps({"event":"evaluation_start","model":name}),flush=True)
    lat=latency_ms(model,cfg.board_size); evaluation=evaluate_vs_random(model,cfg)
    for row in rows: row["inference_latency_ms"]=lat
    (out/"config.json").write_text(json.dumps(cfg.dict(),indent=2)); torch.save(model.state_dict(),out/"model.pt")
    with (out/"metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    runtime={"python":platform.python_version(),"pytorch":torch.__version__,"device":"cpu","inference_latency_ms":lat}; (out/"runtime.json").write_text(json.dumps(runtime,indent=2))
    inspect_tactics(model,out/"tactical_inspection.json")
    (out/"summary.md").write_text(f"# {name} summary\n\nCompleted {cfg.training_iterations} training iterations, {cfg.training_iterations*cfg.self_play_games} self-play games, and {total_samples} generated positions. Batch-1 latency: {lat:.3f} ms. Evaluation: {evaluation}.\n\nTactical examples are inspection-only. Attention weights, where present, are not causal explanations.\n")
    print(json.dumps({"event":"run_complete","model":name,"output":str(out),"positions_total":total_samples,"inference_latency_ms":round(lat,3),"evaluation":evaluation}),flush=True)
    return rows
