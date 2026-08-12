import json
import numpy as np
import torch
from .game import GomokuState
from .mcts import search

def evaluate_vs_random(model,cfg,games=2):
    wins=draws=0
    for g in range(games):
        state=GomokuState.initial(cfg.board_size,cfg.win_length); model_player=1 if g%2==0 else -1
        while not state.terminal():
            if state.to_play==model_player: action=int(search(model,state,cfg.mcts_playouts,cfg.c_puct,0).argmax())
            else: action=int(np.random.choice(state.legal_actions()))
            state=state.play(action)
        wins+=state.winner()==model_player; draws+=state.winner()==0
    return {"evaluation_games":games,"wins_vs_random":int(wins),"draws":int(draws)}

def tactical_positions():
    # Labels are inspection metadata only and never become network input features.
    specs=[("complete horizontal four",(7,0,8,1,9,2),10),("block vertical four",(0,8,1,14,6,20),26),("complete diagonal four",(0,1,7,2,14,3),21)]
    out=[]
    for name,moves,label in specs:
        s=GomokuState.initial()
        for a in moves: s=s.play(a)
        out.append((name,s,label))
    return out

def inspect_tactics(model,path):
    report=[]; model.eval()
    for name,state,label in tactical_positions():
        x=torch.from_numpy(state.features()).unsqueeze(0)
        with torch.no_grad(): logits,value=model(x); probs=torch.softmax(logits.masked_fill(torch.tensor(state.board.reshape(-1)!=0)[None],-torch.inf),1)[0]
        top=torch.topk(probs,3); item={"name":name,"inspection_label":label,"top_moves":[{"action":int(a),"probability":float(p)} for p,a in zip(top.values,top.indices)],"value":float(value)}
        if hasattr(model,"last_attention"):
            item["attention_mean_by_relation"]={k:float(v.mean()) for k,v in model.last_attention.items()}
        if getattr(model,"last_semantic_attention",None) is not None: item["semantic_attention"]=[float(x) for x in model.last_semantic_attention]
        report.append(item)
    path.write_text(json.dumps({"disclaimer":"Attention weights are descriptive inspection values, not causal explanations.","positions":report},indent=2))
