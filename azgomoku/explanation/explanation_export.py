"""CLI and API for post-MCTS state-specific SVG decision evidence."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from azgomoku.game import GomokuState
from azgomoku.mcts import search
from models.han import HAN
from models.rgat import RGAT
from models.rgcn import RGCN

from .explanation_schema import make_document
from .mcts_trace import extract_mcts_trace
from .model_evidence import collect_model_evidence
from .rendering import render_board_svg, render_decision_svg, render_graph_svg


MODEL_CLASSES={"rgcn":RGCN,"rgat":RGAT,"han":HAN}


def write_svgs(document,output_dir):
    """Render only from structured evidence; no model or MCTS is required."""
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    outputs={"board.svg":render_board_svg(document),"graph.svg":render_graph_svg(document),"decision.svg":render_decision_svg(document)}
    for name,content in outputs.items(): (output_dir/name).write_text(content,encoding="utf-8")
    return outputs


def explain_decision(state,model,selected_move,mcts_root=None,output_dir=None,top_k_candidates=5,top_k_edges=12,checkpoint=None,playouts=None,mcts_search_ms=None,mode="svg"):
    """Explain the immutable pre-move root after search, with one evidence forward."""
    if mode=="off": return None
    if mode!="svg": raise ValueError("mode must be 'off' or 'svg'")
    selected_move=int(selected_move)
    if selected_move not in set(map(int,state.legal_actions())): raise ValueError(f"selected move {selected_move} is illegal")
    total_start=time.perf_counter(); evidence_start=time.perf_counter(); evidence=collect_model_evidence(state,model,selected_move); evidence_ms=(time.perf_counter()-evidence_start)*1000
    document=make_document(state,evidence["model_type"],checkpoint,selected_move,top_k_edges)
    document["network"]=evidence["network"]; document["graph_evidence"]=evidence["graph_evidence"]; document["semantic_attention"]=evidence["semantic_attention"]; document["limitations"].extend(evidence["limitations"])
    document["mcts"]=extract_mcts_trace(mcts_root,selected_move,state.size,evidence["network"]["raw_policy_priors"],top_k_candidates,playouts)
    document["runtime_ms"]={"mcts_search_ms":mcts_search_ms,"evidence_forward_ms":evidence_ms,"json_export_ms":None,"svg_render_ms":None,"total_explanation_ms":None}
    if output_dir is not None:
        output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
        json_start=time.perf_counter(); json.dumps(document,indent=2,sort_keys=True); json_ms=(time.perf_counter()-json_start)*1000
        svg_start=time.perf_counter(); write_svgs(document,output_dir); svg_ms=(time.perf_counter()-svg_start)*1000
        document["runtime_ms"].update({"json_export_ms":json_ms,"svg_render_ms":svg_ms,"total_explanation_ms":(time.perf_counter()-total_start)*1000})
        (output_dir/"explanation.json").write_text(json.dumps(document,indent=2,sort_keys=True),encoding="utf-8")
    return document


def load_state(path):
    data=json.loads(Path(path).read_text(encoding="utf-8")); return GomokuState(np.asarray(data["board"],dtype=np.int8),int(data.get("to_play",1)),int(data.get("last_move",-1)),int(data.get("win_length",4)))


def load_model(name,checkpoint,board_size,hidden_dim=64,attention_heads=4):
    model=MODEL_CLASSES[name](board_size=board_size,hidden_dim=hidden_dim,attention_heads=attention_heads); model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)); model.eval(); return model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace",type=Path,help="render a saved explanation JSON without model or MCTS")
    parser.add_argument("--model",choices=tuple(MODEL_CLASSES)); parser.add_argument("--checkpoint",type=Path); parser.add_argument("--state",type=Path)
    parser.add_argument("--output",type=Path,required=True); parser.add_argument("--selected-move",type=int)
    parser.add_argument("--mcts-playouts",type=int,default=50); parser.add_argument("--top-k-candidates",type=int,default=5); parser.add_argument("--top-k-edges",type=int,default=12); parser.add_argument("--seed",type=int,default=7)
    args=parser.parse_args()
    if args.trace:
        document=json.loads(args.trace.read_text(encoding="utf-8")); write_svgs(document,args.output); print((args.output/"decision.svg").resolve()); return
    if not all((args.model,args.checkpoint,args.state)): parser.error("primary mode requires --model, --checkpoint, and --state")
    torch.manual_seed(args.seed); np.random.seed(args.seed); state=load_state(args.state); model=load_model(args.model,args.checkpoint,state.size)
    search_start=time.perf_counter(); policy,root=search(model,state,playouts=args.mcts_playouts,temperature=1.0,return_root=True); search_ms=(time.perf_counter()-search_start)*1000
    selected=int(policy.argmax()) if args.selected_move is None else args.selected_move
    explain_decision(state,model,selected,root,args.output,args.top_k_candidates,args.top_k_edges,args.checkpoint,args.mcts_playouts,search_ms,"svg")
    print((args.output/"decision.svg").resolve())


if __name__=="__main__": main()
