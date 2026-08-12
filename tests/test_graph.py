from azgomoku.graph import cell_graph,metapath_edges
def test_cached_and_symmetric():
    a=cell_graph(6); assert a is cell_graph(6) and len(a)==4
    for e in a:
        pairs=set(map(tuple,e.t().tolist())); assert all((b,a) in pairs for a,b in pairs)
    assert len(metapath_edges(6))==4
