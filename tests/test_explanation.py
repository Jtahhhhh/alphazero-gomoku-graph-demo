import json
import numpy as np
import torch

from azgomoku.explanation.explanation_export import explain_decision, write_svgs
from azgomoku.explanation.mcts_trace import extract_mcts_trace
from azgomoku.explanation.rendering import render_decision_svg, render_graph_svg, select_render_edges
from azgomoku.game import GomokuState
from azgomoku.mcts import Node, search
from models.rgat import RGAT
from models.rgcn import RGCN
from investigation.e3b_graph import structural_edges


def state3(): return GomokuState(np.array([[1,-1,0],[0,1,0],[0,0,-1]],dtype=np.int8),1,8,3)


def test_selected_action_orientation_and_pre_move_state(tmp_path):
    state=state3(); before=state.board.copy(); result=explain_decision(state,RGCN(board_size=3,hidden_dim=8),5,output_dir=tmp_path)
    assert result["selected_move"]=={"action":5,"row":1,"col":2}
    assert result["state"]["board"][1][2]==0 and np.array_equal(state.board,before)
    svg=(tmp_path/"board.svg").read_text(); assert 'data-node-row="1" data-node-col="2" data-action="5" data-role="selected_move"' in svg
    assert (tmp_path/"decision.svg").is_file() and not (tmp_path/"graph.png").exists()


def test_rgcn_has_structure_but_no_attention():
    result=explain_decision(state3(),RGCN(board_size=3,hidden_dim=8),5)
    assert not result["graph_evidence"]["attention_available"]
    assert result["graph_evidence"]["edges"] and all(x["attention"] is None for x in result["graph_evidence"]["edges"])


def test_rgat_values_and_head_mean_are_exact():
    model=RGAT(board_size=3,hidden_dim=8,attention_heads=2); result=explain_decision(state3(),model,5); first=result["graph_evidence"]["edges"][0]
    x=torch.from_numpy(state3().features()).unsqueeze(0)
    with torch.no_grad(): _,_,raw=model(x,return_evidence=True)
    expected=[float(v) for v in raw["relation_attention"][first["relation"]][0,0].tolist()]
    assert first["head_attention"]==expected and first["attention"]==sum(expected)/len(expected)


def test_mcts_root_statistics():
    root=Node(); root.children={2:Node(.6),5:Node(.4)}; root.children[2].n,root.children[2].w=3,1.5; root.children[5].n,root.children[5].w=7,4.2
    trace=extract_mcts_trace(root,5,3,[0,0,.55,0,0,.45,0,0,0],top_k=2,playouts=10)
    assert trace["selected"]["visits"]==7 and trace["selected"]["q"]==4.2/7 and trace["selected"]["pi"]==.7
    assert trace["selected"]["raw_policy_prior"]==.45 and trace["selected"]["search_prior"]==.4
    assert trace["mcts_value_convention_version"]==2
    assert trace["q_perspective"]=="player_who_selects_action_at_parent"


def test_deterministic_filter_and_trace_only_render(tmp_path):
    torch.manual_seed(11); document=explain_decision(state3(),RGAT(board_size=3,hidden_dim=8),5,top_k_edges=3)
    assert len(select_render_edges(document))==3
    assert render_graph_svg(document)==render_graph_svg(document)
    saved=tmp_path/"saved.json"; saved.write_text(json.dumps(document)); out=tmp_path/"rendered"; write_svgs(json.loads(saved.read_text()),out)
    assert (out/"decision.svg").read_text()==render_decision_svg(document)


def test_optional_knowledge_svg_is_registered_beside_existing_three(tmp_path):
    state=state3(); model=RGAT(board_size=3,hidden_dim=8,attention_heads=2)
    document=explain_decision(state,model,5)
    record={
        "state_id":"knowledge-test",
        "state":{"board_size":3,"win_length":3,"current_player":1,"last_move":8,"board":state.board.tolist()},
        "solver":{"status":"exact_partial","optimal_actions_complete":False},
        "valid_proofs":[{"action":3,"concepts":["mandatory_block"],"critical_cells":[3,4,5],"critical_relations":["horizontal"],"windows":[[3,4,5]]}],
    }
    payload={
        "record":record,
        "rgat_edges":document["graph_evidence"]["edges"],
        "structural_edges":structural_edges(3),
        "metrics":{"attention_collapse_flag":1,"attention_normalized_entropy":1,"attention_head_diversity":0,"attention_topology_correlation":1,"graph_critical_mass":0},
        "graph_gate":{"passed":True,"d4_proof_roundtrips":8},
        "artifact_version":2,
        "decision":{
            "selected_move":document["selected_move"],
            "actor":document["model"],
            "attention_source":{"model":document["model"],"relationship_to_actor":"actor"},
        },
    }
    outputs=write_svgs(document,tmp_path,knowledge=payload)
    assert set(outputs)=={"board.svg","graph.svg","decision.svg","knowledge.svg"}
    svg=(tmp_path/"knowledge.svg").read_text(encoding="utf-8")
    assert "PARTIAL · PARTIAL KNOWLEDGE" in svg
    assert 'data-role="proof-action-marker" data-action="3"' in svg
    assert 'data-layer="mcts-selected" data-board="tactic" data-action="5"' in svg
    assert "PROOF #1" in svg and "MCTS SELECTED action=5" in svg


def test_normal_mcts_forces_evidence_off():
    class Spy(RGCN):
        def __init__(self): super().__init__(board_size=3,hidden_dim=8); self.flags=[]
        def forward(self,x,return_evidence=False): self.flags.append(return_evidence); return super().forward(x,return_evidence)
    model=Spy(); _,root=search(model,GomokuState.initial(3,3),playouts=2,return_root=True)
    assert model.flags and not any(model.flags)
    assert sum(child.n for child in root.children.values())==2


def test_one_explanation_forward_and_off_mode():
    class Spy(RGAT):
        def __init__(self): super().__init__(board_size=3,hidden_dim=8); self.flags=[]
        def forward(self,x,return_evidence=False): self.flags.append(return_evidence); return super().forward(x,return_evidence)
    model=Spy(); assert explain_decision(state3(),model,5,mode="off") is None and model.flags==[]
    explain_decision(state3(),model,5); assert model.flags==[True]
