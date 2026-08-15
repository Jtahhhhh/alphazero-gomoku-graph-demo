import json

import numpy as np
import pytest
import torch

from azgomoku.explanation.game_export import _resolve_initial_state, export_game, game_seed, select_action
from azgomoku.mcts import Node
from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.game import GomokuState
from models.rgcn import RGCN


def near_terminal(): return GomokuState(np.asarray([[1,-1,1],[-1,1,-1],[0,0,0]],dtype=np.int8),to_play=1,last_move=5,win_length=3)


def test_cli_board_and_win_length_create_the_requested_rules(tmp_path):
    default = _resolve_initial_state()
    assert default.size == 6 and default.win_length == 4
    state = _resolve_initial_state(board_size=10, win_length=5)
    assert state.size == 10 and state.win_length == 5
    with pytest.raises(ValueError, match="between 2"):
        _resolve_initial_state(board_size=10, win_length=11)

    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps({
        "board": np.zeros((10, 10), dtype=int).tolist(),
        "to_play": 1,
        "last_move": -1,
        "win_length": 5,
    }), encoding="utf-8")
    loaded = _resolve_initial_state(state_path, board_size=10, win_length=5)
    assert loaded.size == 10 and loaded.win_length == 5
    with pytest.raises(ValueError, match="board-size"):
        _resolve_initial_state(state_path, board_size=15, win_length=5)


def test_self_game_exports_every_model_move_and_continuous_state_ids(tmp_path):
    torch.manual_seed(4); initial=near_terminal(); manifest=export_game(RGCN(board_size=3,hidden_dim=8),tmp_path,initial_state=initial,opponent="self",mcts_playouts=2,mode="eval",base_seed=4)
    assert manifest["terminal"] and manifest["moves"]
    assert all(move["evidence_available"] for move in manifest["moves"])
    assert all(
        (tmp_path / move["artifact_dir"] / name).is_file()
        for move in manifest["moves"]
        for name in ("board.svg", "graph.svg", "decision.svg")
    )
    assert all(manifest["moves"][i]["next_state_id"]==manifest["moves"][i+1]["state_id"] for i in range(len(manifest["moves"])-1))
    assert manifest["moves"][0]["state_id"]==state_identifier(initial)
    assert (tmp_path/"game.json").is_file()


def test_random_opponent_has_no_fake_evidence_and_preserves_last_move(tmp_path):
    initial=near_terminal(); manifest=export_game(RGCN(board_size=3,hidden_dim=8),tmp_path,initial_state=initial,opponent="random",model_player=-1,mcts_playouts=1,mode="eval",base_seed=3,max_moves=2)
    first=manifest["moves"][0]
    assert first["actor"]["type"]=="random" and not first["evidence_available"] and first["artifact_dir"] is None
    assert manifest["final_state"]["last_move"]==manifest["moves"][-1]["action"]


def test_second_model_is_used_for_opposing_player(tmp_path):
    first=RGCN(board_size=3,hidden_dim=8); second=RGCN(board_size=3,hidden_dim=8)
    manifest=export_game(first,tmp_path,initial_state=near_terminal(),opponent="model",opponent_model=second,model_player=1,mcts_playouts=1,mode="eval",max_moves=2)
    assert all(move["evidence_available"] for move in manifest["moves"])
    assert {move["player"] for move in manifest["moves"]}<={1,-1}


def root_with_visits(*counts):
    root=Node(); root.children={i:Node() for i in range(len(counts))}
    for i,count in enumerate(counts): root.children[i].n=count
    return root


def test_eval_is_greedy_and_temperature_independent():
    root=root_with_visits(2,7,4)
    for seed in range(5):
        assert select_action(root,3,mode="eval",temperature=10.0,rng=np.random.default_rng(seed))==1


def test_data_selection_is_reproducible_per_game_and_diverse_across_games():
    root=root_with_visits(4,3,2,1); actions=[]
    for game_index in range(20):
        seed=game_seed(17,game_index)
        first=select_action(root,4,mode="data",temperature=1.0,rng=np.random.default_rng(seed))
        second=select_action(root,4,mode="data",temperature=1.0,rng=np.random.default_rng(seed))
        assert first==second
        actions.append(first)
    assert len(set(actions))>=3
    assert [game_seed(17,i) for i in range(5)]!=[game_seed(18,i) for i in range(5)]


def test_manifest_records_data_mode_and_game_seed(tmp_path):
    manifest=export_game(RGCN(board_size=3,hidden_dim=8),tmp_path,initial_state=near_terminal(),opponent="self",mcts_playouts=2,mode="data",base_seed=9,game_index=3,max_moves=1)
    assert manifest["mode"]=="data" and manifest["settings"]["mode"]=="data"
    assert manifest["seed"]==game_seed(9,3)
