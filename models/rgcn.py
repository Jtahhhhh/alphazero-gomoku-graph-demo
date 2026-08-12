import torch
from torch import nn
from azgomoku.graph import cell_graph
from .common import PolicyValueHeads

class RelGraphConv(nn.Module):
    def __init__(self,inp,out):
        super().__init__(); self.self_lin=nn.Linear(inp,out); self.rel=nn.ModuleList(nn.Linear(inp,out,bias=False) for _ in range(4))
    def forward(self,x,edges):
        out=self.self_lin(x)
        for lin,e in zip(self.rel,edges):
            src,dst=e; msg=lin(x[:,src]); agg=torch.zeros_like(out); agg.index_add_(1,dst,msg)
            deg=torch.zeros(x.shape[1],device=x.device).index_add_(0,dst,torch.ones_like(dst,dtype=x.dtype)); out=out+agg/deg.clamp_min(1)[None,:,None]
        return torch.relu(out)

class RGCN(nn.Module):
    """Two-layer relational GCN over one row-major Cell node per action."""
    def __init__(self,board_size=6,hidden_dim=64,**_):
        super().__init__(); self.board_size=board_size
        for i,e in enumerate(cell_graph(board_size)): self.register_buffer(f"edge_{i}",e)
        self.layers=nn.ModuleList((RelGraphConv(6,hidden_dim),RelGraphConv(hidden_dim,hidden_dim))); self.heads=PolicyValueHeads(hidden_dim)
    def forward(self,x):
        h=x.flatten(2).transpose(1,2); edges=tuple(getattr(self,f"edge_{i}") for i in range(4))
        for layer in self.layers: h=layer(h,edges)
        return self.heads(h)
