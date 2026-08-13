from azgomoku.graph import cell_edge_records,cell_graph
def test_cached_and_symmetric():
    a=cell_graph(6); assert a is cell_graph(6) and len(a)==4
    for e in a:
        pairs=set(map(tuple,e.t().tolist())); assert all((b,a) in pairs for a,b in pairs)
def test_stable_edge_identities():
    cell=cell_edge_records(4)
    assert len({x["edge_id"] for x in cell})==len(cell)
