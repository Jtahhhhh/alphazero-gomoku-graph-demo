import math
import torch
from torch import nn
from azgomoku.graph import cell_graph,RELATIONS
from .common import PolicyValueHeads

class RelAttention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__(); assert dim%heads==0; self.heads=heads; self.d=dim//heads
        self.q=nn.ModuleList(nn.Linear(dim,dim,bias=False) for _ in range(4)); self.k=nn.ModuleList(nn.Linear(dim,dim,bias=False) for _ in range(4)); self.v=nn.ModuleList(nn.Linear(dim,dim,bias=False) for _ in range(4)); self.out=nn.Linear(dim,dim)
    def forward(self,x,edges,return_evidence=False):
        total=torch.zeros_like(x); recorded={} if return_evidence else None
        for name,q,k,v,e in zip(RELATIONS,self.q,self.k,self.v,edges):
            src,dst=e; qq=q(x[:,dst]).view(x.shape[0],-1,self.heads,self.d); kk=k(x[:,src]).view(x.shape[0],-1,self.heads,self.d); vv=v(x[:,src]).view(x.shape[0],-1,self.heads,self.d)
            score=(qq*kk).sum(-1)/math.sqrt(self.d); alpha=torch.zeros_like(score)
            for node in range(x.shape[1]):
                mask=dst==node
                if mask.any(): alpha[:,mask]=torch.softmax(score[:,mask],dim=1)
            msg=(alpha[...,None]*vv).reshape(x.shape[0],-1,self.heads*self.d); agg=torch.zeros_like(x); agg.index_add_(1,dst,msg); total+=agg
            if return_evidence: recorded[name]=alpha.detach()
        return torch.relu(x+self.out(total/4)),recorded

class RGAT(nn.Module):
    """Two-layer relational attention encoder sharing R-GCN's Cell graph."""
    def __init__(self,board_size=6,hidden_dim=64,attention_heads=4,**_):
        super().__init__(); self.board_size=board_size; self.input=nn.Linear(6,hidden_dim)
        for i,e in enumerate(cell_graph(board_size)): self.register_buffer(f"edge_{i}",e)
        self.layers=nn.ModuleList(RelAttention(hidden_dim,attention_heads) for _ in range(2)); self.heads=PolicyValueHeads(hidden_dim)
    def forward(self,x,return_evidence=False):
        h=torch.relu(self.input(x.flatten(2).transpose(1,2))); edges=tuple(getattr(self,f"edge_{i}") for i in range(4))
        recorded=None
        for index,layer in enumerate(self.layers): h,recorded=layer(h,edges,return_evidence and index==len(self.layers)-1)
        outputs=self.heads(h)
        evidence={"attention_available":True,"relation_attention":recorded,"layer":"final"}
        return (*outputs,evidence) if return_evidence else outputs
