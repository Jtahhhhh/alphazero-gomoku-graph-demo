"""Export state-specific decision evidence for every move of one complete game."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from azgomoku.game import GomokuState
from azgomoku.mcts import search

from .explanation_export import MODEL_CLASSES, explain_decision, load_model, load_state
from .explanation_schema import state_identifier


def _model_spec(model, checkpoint):
    return {"type":model.__class__.__name__.lower(),"checkpoint":str(checkpoint) if checkpoint else None}


def export_game(
    model,
    output_dir,
    *,
    checkpoint=None,
    initial_state=None,
    opponent="self",
    opponent_model=None,
    opponent_checkpoint=None,
    model_player=1,
    mcts_playouts=50,
    temperature=0.0,
    top_k_candidates=5,
    top_k_edges=12,
    seed=7,
    max_moves=None,
):
    """Play and export one game; evidence is collected once after each model search."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    state=initial_state or GomokuState.initial(model.board_size,4)
    rng=np.random.default_rng(seed); torch.manual_seed(seed)
    manifest={
        "schema_version":1,"artifact_type":"full_game_decision_evidence","seed":seed,
        "board_size":state.size,"win_length":state.win_length,"initial_state_id":state_identifier(state),
        "settings":{"opponent":opponent,"model_player":model_player,"mcts_playouts":mcts_playouts,"temperature":temperature,"top_k_candidates":top_k_candidates,"top_k_edges":top_k_edges,"max_moves":max_moves},
        "players":{"1":None,"-1":None},"moves":[],"terminal":False,"winner":None,"truncated":False,
    }
    primary=_model_spec(model,checkpoint)
    if opponent=="self": manifest["players"]={"1":primary,"-1":primary}
    elif opponent=="random":
        manifest["players"][str(model_player)]=primary; manifest["players"][str(-model_player)]={"type":"random","checkpoint":None}
    elif opponent=="model":
        if opponent_model is None: raise ValueError("opponent_model is required when opponent='model'")
        manifest["players"][str(model_player)]=primary; manifest["players"][str(-model_player)]=_model_spec(opponent_model,opponent_checkpoint)
    else: raise ValueError("opponent must be 'self', 'random', or 'model'")

    ply=1
    while not state.terminal() and (max_moves is None or ply<=max_moves):
        before_id=state_identifier(state); player=int(state.to_play); actor=manifest["players"][str(player)]
        active_model=None
        if opponent=="self" or player==model_player: active_model=model
        elif opponent=="model": active_model=opponent_model
        move_dir=output_dir/f"move_{ply:03d}"
        if active_model is None:
            legal=list(map(int,state.legal_actions())); action=int(rng.choice(legal)); search_ms=None; document=None
        else:
            start=time.perf_counter(); pi,root=search(active_model,state,playouts=mcts_playouts,temperature=temperature,return_root=True); search_ms=(time.perf_counter()-start)*1000
            action=int(pi.argmax()); document=explain_decision(state,active_model,action,root,move_dir,top_k_candidates,top_k_edges,actor.get("checkpoint"),mcts_playouts,search_ms,"svg")
        next_state=state.play(action)
        item={"ply":ply,"player":player,"actor":actor,"state_id":before_id,"action":action,"row":action//state.size,"col":action%state.size,"next_state_id":state_identifier(next_state),"search_ms":search_ms,"evidence_available":document is not None,"artifact_dir":move_dir.name if document is not None else None}
        manifest["moves"].append(item); state=next_state; ply+=1
        print(json.dumps({"event":"move_complete",**item}),flush=True)
    manifest["terminal"]=bool(state.terminal()); manifest["winner"]=int(state.winner()) if state.terminal() else None; manifest["truncated"]=not state.terminal()
    manifest["final_state"]={"state_id":state_identifier(state),"to_play":int(state.to_play),"last_move":int(state.last_move),"board":state.board.astype(int).tolist()}
    (output_dir/"game.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps({"event":"game_complete","moves":len(manifest["moves"]),"terminal":manifest["terminal"],"winner":manifest["winner"],"output":str(output_dir/"game.json")}),flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",choices=tuple(MODEL_CLASSES),required=True); parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--opponent",choices=("self","random","model"),default="self"); parser.add_argument("--opponent-model",choices=tuple(MODEL_CLASSES)); parser.add_argument("--opponent-checkpoint",type=Path)
    parser.add_argument("--model-player",type=int,choices=(-1,1),default=1); parser.add_argument("--state",type=Path); parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mcts-playouts",type=int,default=50); parser.add_argument("--temperature",type=float,default=0.0); parser.add_argument("--top-k-candidates",type=int,default=5); parser.add_argument("--top-k-edges",type=int,default=12); parser.add_argument("--seed",type=int,default=7); parser.add_argument("--max-moves",type=int)
    args=parser.parse_args(); state=load_state(args.state) if args.state else GomokuState.initial(); model=load_model(args.model,args.checkpoint,state.size)
    opponent_model=None
    if args.opponent=="model":
        if not args.opponent_model or not args.opponent_checkpoint: parser.error("--opponent model requires --opponent-model and --opponent-checkpoint")
        opponent_model=load_model(args.opponent_model,args.opponent_checkpoint,state.size)
    export_game(model,args.output,checkpoint=args.checkpoint,initial_state=state,opponent=args.opponent,opponent_model=opponent_model,opponent_checkpoint=args.opponent_checkpoint,model_player=args.model_player,mcts_playouts=args.mcts_playouts,temperature=args.temperature,top_k_candidates=args.top_k_candidates,top_k_edges=args.top_k_edges,seed=args.seed,max_moves=args.max_moves)


if __name__=="__main__": main()
