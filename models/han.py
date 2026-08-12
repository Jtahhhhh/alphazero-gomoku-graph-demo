import math
import torch
from torch import nn
from azgomoku.graph import metapath_edges,line_memberships,RELATIONS
from .common import PolicyValueHeads

class MetaPathAttention(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.q=nn.Linear(dim,dim,bias=False); self.k=nn.Linear(dim,dim,bias=False); self.v=nn.Linear(dim,dim,bias=False)
    def forward(self,x,e,return_evidence=False):
        src,dst=e; score=(self.q(x[:,dst])*self.k(x[:,src])).sum(-1)/math.sqrt(x.shape[-1]); alpha=torch.zeros_like(score)
        for node in range(x.shape[1]):
            mask=dst==node
            if mask.any(): alpha[:,mask]=torch.softmax(score[:,mask],dim=1)
        msg=alpha[...,None]*self.v(x[:,src]); out=torch.zeros_like(x); out.index_add_(1,dst,msg)
        return torch.relu(out),alpha.detach() if return_evidence else None

class HAN(nn.Module):
    """HAN over Cell--directional Line--Cell meta-paths.

    Structural Line nodes are defined by cached memberships. Each meta-path edge is
    derived from two incidence hops through one such Line node.
    """
    def __init__(self,board_size=6,hidden_dim=64,**_):
        super().__init__(); self.board_size=board_size; self.lines=line_memberships(board_size); self.input=nn.Linear(6,hidden_dim)
        for i,e in enumerate(metapath_edges(board_size)): self.register_buffer(f"meta_{i}",e)
        self.node_attention=nn.ModuleList(MetaPathAttention(hidden_dim) for _ in range(4))
        self.semantic_logits=nn.Parameter(torch.zeros(4)); self.heads=PolicyValueHeads(hidden_dim)
    def forward(self,x,return_evidence=False):
        h=torch.relu(self.input(x.flatten(2).transpose(1,2))); views=[]; att={} if return_evidence else None
        for i,(name,layer) in enumerate(zip(RELATIONS,self.node_attention)):
            view,a=layer(h,getattr(self,f"meta_{i}"),return_evidence); views.append(view)
            if return_evidence: att[name]=a
        semantic=torch.softmax(self.semantic_logits,0); h=sum(semantic[i]*views[i] for i in range(4))
        outputs=self.heads(h)
        evidence={"attention_available":True,"node_attention":att,"semantic_attention":semantic.detach()}
        return (*outputs,evidence) if return_evidence else outputs
