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
def cell_edge_records(size):
    return tuple({"edge_id":f"{relation}:{int(source)}:{int(target)}","source":int(source),"target":int(target),"relation":relation} for relation,group in zip(RELATIONS,cell_graph(size)) for source,target in group.t().tolist())


def structural_edges(size: int) -> list[dict]:
    """Return deterministic cell-graph edges with the structural attention baseline."""
    records = cell_edge_records(size)
    indegree = {}
    for item in records:
        key = (item["relation"], item["target"])
        indegree[key] = indegree.get(key, 0) + 1
    return [
        {
            "edge_id": item["edge_id"],
            "relation": item["relation"],
            "source": {"action": item["source"]},
            "target": {"action": item["target"]},
            "attention": 1.0 / indegree[(item["relation"], item["target"])],
        }
        for item in records
    ]
