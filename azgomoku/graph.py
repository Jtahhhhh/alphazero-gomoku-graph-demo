from functools import lru_cache
import torch
RELATIONS=("horizontal","vertical","diagonal_down","diagonal_up")
@lru_cache(maxsize=None)
def cell_graph(size):
    edges=[]
    for dr,dc in ((0,1),(1,0),(1,1),(1,-1)):
        pairs=[]
        for r in range(size):
            for c in range(size):
                nr,nc=r+dr,c+dc
                if 0<=nr<size and 0<=nc<size:
                    a,b=r*size+c,nr*size+nc; pairs.extend(((a,b),(b,a)))
        edges.append(torch.tensor(pairs,dtype=torch.long).t().contiguous())
    return tuple(edges)
@lru_cache(maxsize=None)
def line_memberships(size):
    fs=[[[r*size+c for c in range(size)] for r in range(size)],[[r*size+c for r in range(size)] for c in range(size)],[[r*size+c for r in range(size) for c in range(size) if r-c==d] for d in range(-(size-1),size)],[[r*size+c for r in range(size) for c in range(size) if r+c==s] for s in range(2*size-1)]]
    return tuple(tuple(tuple(x) for x in f if len(x)>1) for f in fs)
@lru_cache(maxsize=None)
def metapath_edges(size):
    out=[]
    for family in line_memberships(size):
        pairs=[]
        for line in family: pairs.extend((a,b) for a in line for b in line)
        out.append(torch.tensor(pairs,dtype=torch.long).t().contiguous())
    return tuple(out)
