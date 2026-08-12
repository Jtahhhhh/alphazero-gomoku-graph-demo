import numpy as np
import torch
from azgomoku.game import GomokuState
from models.common import masked_policy

def test_legal_and_winner():
    s=GomokuState.initial()
    for a in (0,6,1,7,2,8,3): s=s.play(a)
    assert s.winner()==1 and s.terminal() and 0 not in s.legal_actions()
def test_action_mapping():
    s=GomokuState.initial().play(17); assert s.board[2,5]==1 and s.features()[2].reshape(-1)[17]==1
def test_policy_mask():
    p=masked_policy(torch.zeros(1,36),torch.tensor([[False]+[True]*35])); assert p[0,0]==0 and torch.isclose(p.sum(),torch.tensor(1.))
