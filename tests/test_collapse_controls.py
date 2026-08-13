import numpy as np
import torch

from azgomoku.config import Config
from azgomoku.game import GomokuState
from azgomoku.mcts import search
from azgomoku.training import self_play, symmetry_augment
from models.rgcn import RGCN


def test_symmetry_augmentation_preserves_feature_policy_alignment():
    features=np.zeros((6,3,3),dtype=np.float32); features[0,0,1]=1
    policy=np.zeros(9,dtype=np.float32); policy[1]=1
    variants=symmetry_augment(features,policy)
    assert len(variants)==8
    for x,pi in variants:
        assert np.isclose(pi.sum(),1)
        assert int(np.argmax(x[0]))==int(np.argmax(pi))
    assert len({int(np.argmax(pi)) for _,pi in variants})==4


def test_root_dirichlet_noise_changes_priors_but_keeps_distribution():
    torch.manual_seed(1); np.random.seed(1); model=RGCN(board_size=3,hidden_dim=8)
    state=GomokuState.initial(3,3)
    _,plain=search(model,state,playouts=0,return_root=True)
    np.random.seed(1); _,noisy=search(model,state,playouts=0,return_root=True,dirichlet_alpha=.3,dirichlet_fraction=.25)
    p=np.array([plain.children[a].prior for a in sorted(plain.children)])
    q=np.array([noisy.children[a].prior for a in sorted(noisy.children)])
    assert np.isclose(q.sum(),1) and not np.allclose(p,q)


def test_self_play_augments_replay_and_reports_opening_diagnostics():
    np.random.seed(2); torch.manual_seed(2); model=RGCN(board_size=3,hidden_dim=8)
    cfg=Config(board_size=3,win_length=3,mcts_playouts=1,self_play_games=1,symmetry_augmentation=True,opening_temperature_moves=2,dirichlet_fraction=.25)
    samples,metrics=self_play(model,cfg)
    assert len(samples)%8==0
    assert metrics["opening_unique_actions"]==1
    assert 0<=metrics["opening_corner_mass"]<=1
    assert 0<=metrics["opening_edge_mass"]<=1
