import torch
from torch import nn
class PolicyValueHeads(nn.Module):
    def __init__(self,hidden):
        super().__init__(); self.policy=nn.Linear(hidden,1); self.value=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1),nn.Tanh())
    def forward(self,nodes): return self.policy(nodes).squeeze(-1),self.value(nodes.mean(1)).squeeze(-1)
def masked_policy(logits,legal_mask): return torch.softmax(logits.masked_fill(~legal_mask,torch.finfo(logits.dtype).min),dim=-1)
