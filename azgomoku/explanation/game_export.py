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


def game_seed(base_seed, game_index):
    """Derive a stable, independent NumPy seed for one game."""
    return int(np.random.SeedSequence([int(base_seed),int(game_index)]).generate_state(1,dtype=np.uint64)[0])


def _resolve_initial_state(state_path=None, board_size=None, win_length=None):
    """Build the CLI root state without silently changing board rules."""
    if state_path is not None:
        state = load_state(state_path)
        if board_size is not None and int(board_size) != state.size:
            raise ValueError("--board-size does not match --state")
        if win_length is not None and int(win_length) != state.win_length:
            raise ValueError("--win-length does not match --state")
        return state
    size = 6 if board_size is None else int(board_size)
    target = 4 if win_length is None else int(win_length)
    if size < 2:
        raise ValueError("--board-size must be at least 2")
    if target < 2 or target > size:
        raise ValueError("--win-length must be between 2 and --board-size")
    return GomokuState.initial(size, target)


def _visit_counts(root, action_count):
    visits=np.zeros(action_count,dtype=np.float64)
    for action,child in root.children.items(): visits[int(action)]=int(child.n)
    return visits


def select_action(root, action_count, *, mode, temperature, rng):
    """Choose the played move from root visits without changing recorded search evidence."""
    if mode not in {"eval","data"}: raise ValueError("mode must be 'eval' or 'data'")
    visits=_visit_counts(root,action_count)
    if visits.sum()<=0: raise ValueError("MCTS root has no visits")
    if temperature<0: raise ValueError("temperature must be non-negative")
    if mode=="eval" or temperature<=1e-6: return int(visits.argmax())
    positive=visits>0; scaled=np.full(action_count,-np.inf,dtype=np.float64)
    scaled[positive]=np.log(visits[positive])/float(temperature); scaled-=np.max(scaled)
    weights=np.exp(scaled); probabilities=weights/weights.sum()
    return int(rng.choice(action_count,p=probabilities))


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
    mode,
    temperature=1.0,
    opening_temperature_moves=10,
    late_temperature=0.0,
    top_k_candidates=5,
    top_k_edges=12,
    base_seed=7,
    game_index=0,
    max_moves=None,
):
    """Play and export one game; evidence is collected once after each model search."""
    if mode not in {"eval","data"}: raise ValueError("mode must be 'eval' or 'data'")
    if temperature<0 or late_temperature<0: raise ValueError("temperatures must be non-negative")
    if opening_temperature_moves<0: raise ValueError("opening_temperature_moves must be non-negative")
    if base_seed<0 or game_index<0: raise ValueError("base_seed and game_index must be non-negative")
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    state=initial_state or GomokuState.initial(model.board_size,4)
    seed=game_seed(base_seed,game_index); rng=np.random.default_rng(seed); torch.manual_seed(seed)
    manifest={
        "schema_version":1,"artifact_type":"full_game_decision_evidence","mode":mode,"base_seed":int(base_seed),"game_index":int(game_index),"seed":seed,
        "board_size":state.size,"win_length":state.win_length,"initial_state_id":state_identifier(state),
        "settings":{"mode":mode,"opponent":opponent,"model_player":model_player,"mcts_playouts":mcts_playouts,"temperature":temperature,"opening_temperature_moves":opening_temperature_moves,"late_temperature":late_temperature,"top_k_candidates":top_k_candidates,"top_k_edges":top_k_edges,"max_moves":max_moves},
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
            start=time.perf_counter(); _,root=search(active_model,state,playouts=mcts_playouts,temperature=1.0,return_root=True); search_ms=(time.perf_counter()-start)*1000
            selection_temperature=temperature if ply<=opening_temperature_moves else late_temperature
            action=select_action(root,state.size**2,mode=mode,temperature=selection_temperature,rng=rng)
            document=explain_decision(state,active_model,action,root,move_dir,top_k_candidates,top_k_edges,actor.get("checkpoint"),mcts_playouts,search_ms,"svg")
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
    parser.add_argument("--board-size",type=int); parser.add_argument("--win-length",type=int)
    parser.add_argument("--mode",choices=("eval","data"),required=True); parser.add_argument("--mcts-playouts",type=int,default=50); parser.add_argument("--temperature",type=float,default=1.0); parser.add_argument("--opening-temperature-moves",type=int,default=10); parser.add_argument("--late-temperature",type=float,default=0.0); parser.add_argument("--top-k-candidates",type=int,default=5); parser.add_argument("--top-k-edges",type=int,default=12); parser.add_argument("--base-seed",type=int,default=7); parser.add_argument("--game-index",type=int,default=0); parser.add_argument("--max-moves",type=int)
    args=parser.parse_args()
    try: state=_resolve_initial_state(args.state,args.board_size,args.win_length)
    except ValueError as error: parser.error(str(error))
    model=load_model(args.model,args.checkpoint,state.size)
    opponent_model=None
    if args.opponent=="model":
        if not args.opponent_model or not args.opponent_checkpoint: parser.error("--opponent model requires --opponent-model and --opponent-checkpoint")
        opponent_model=load_model(args.opponent_model,args.opponent_checkpoint,state.size)
    export_game(model,args.output,checkpoint=args.checkpoint,initial_state=state,opponent=args.opponent,opponent_model=opponent_model,opponent_checkpoint=args.opponent_checkpoint,model_player=args.model_player,mcts_playouts=args.mcts_playouts,mode=args.mode,temperature=args.temperature,opening_temperature_moves=args.opening_temperature_moves,late_temperature=args.late_temperature,top_k_candidates=args.top_k_candidates,top_k_edges=args.top_k_edges,base_seed=args.base_seed,game_index=args.game_index,max_moves=args.max_moves)


if __name__=="__main__": main()
