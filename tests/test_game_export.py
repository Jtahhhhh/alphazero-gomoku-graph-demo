import numpy as np
import torch

from azgomoku.explanation.game_export import export_game
from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.game import GomokuState
from models.rgcn import RGCN


def near_terminal(): return GomokuState(np.asarray([[1,-1,1],[-1,1,-1],[0,0,0]],dtype=np.int8),to_play=1,last_move=5,win_length=3)


def test_self_game_exports_every_model_move_and_continuous_state_ids(tmp_path):
    torch.manual_seed(4); initial=near_terminal(); manifest=export_game(RGCN(board_size=3,hidden_dim=8),tmp_path,initial_state=initial,opponent="self",mcts_playouts=2,seed=4)
    assert manifest["terminal"] and manifest["moves"]
    assert all(move["evidence_available"] for move in manifest["moves"])
    assert all((tmp_path/move["artifact_dir"]/"decision.svg").is_file() for move in manifest["moves"])
    assert all(manifest["moves"][i]["next_state_id"]==manifest["moves"][i+1]["state_id"] for i in range(len(manifest["moves"])-1))
    assert manifest["moves"][0]["state_id"]==state_identifier(initial)
    assert (tmp_path/"game.json").is_file()


def test_random_opponent_has_no_fake_evidence_and_preserves_last_move(tmp_path):
    initial=near_terminal(); manifest=export_game(RGCN(board_size=3,hidden_dim=8),tmp_path,initial_state=initial,opponent="random",model_player=-1,mcts_playouts=1,seed=3,max_moves=2)
    first=manifest["moves"][0]
    assert first["actor"]["type"]=="random" and not first["evidence_available"] and first["artifact_dir"] is None
    assert manifest["final_state"]["last_move"]==manifest["moves"][-1]["action"]


def test_second_model_is_used_for_opposing_player(tmp_path):
    first=RGCN(board_size=3,hidden_dim=8); second=RGCN(board_size=3,hidden_dim=8)
    manifest=export_game(first,tmp_path,initial_state=near_terminal(),opponent="model",opponent_model=second,model_player=1,mcts_playouts=1,max_moves=2)
    assert all(move["evidence_available"] for move in manifest["moves"])
    assert {move["player"] for move in manifest["moves"]}<={1,-1}
