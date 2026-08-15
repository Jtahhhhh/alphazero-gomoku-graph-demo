import numpy as np

from azgomoku.symmetry import canonical_key,d4_roundtrip_self_check,inverse_symmetry,transform_action,transform_flat_proof,transform_state
from investigation.generate_h1_benchmark import generate


def test_d4_roundtrip_canonical_proofs_and_optimal_actions():
    assert d4_roundtrip_self_check()
    records,summary=generate(target=1,seed=17,attempts=50,deadline_ms=100,node_cap=10_000,specs=((6,4,26,28),))
    assert summary["dedup_mode"]=="d4"
    item=records[0]
    from azgomoku.h1_schema import state_from_record
    state=state_from_record(item); proof={"action":1,"critical_cells":[0,1],"critical_relations":["horizontal"],"windows":[[0,1,2,3]]}
    for symmetry in range(8):
        inverse=inverse_symmetry(symmetry,state.size)
        assert canonical_key(transform_state(state,symmetry))==canonical_key(state)
        assert transform_flat_proof(transform_flat_proof(proof,state.size,symmetry),state.size,inverse)==proof
        actions=(1,5,7)
        assert tuple(transform_action(transform_action(a,state.size,symmetry),state.size,inverse) for a in actions)==actions


def test_generator_is_deterministic_midlate_multisize_and_logs_router():
    kwargs=dict(target=3,seed=23,attempts=100,deadline_ms=100,node_cap=20_000,specs=((6,4,26,28),(10,5,25,27),(15,5,25,27)))
    first,summary=generate(**kwargs); second,second_summary=generate(**kwargs)
    assert [item["state_id"] for item in first]==[item["state_id"] for item in second]
    assert summary==second_summary and summary["router"]["total"]==3
    assert {item["state"]["board_size"] for item in first}=={6,10,15}
    assert all(item["provenance"]["ply"]>=25 and item["provenance"]["dedup_mode"]=="d4" for item in first)
    assert all(item["schema_version"]==2 for item in first)


def test_every_written_partial_replays_and_unknown_is_outside_denominator():
    from azgomoku.h1_schema import validate_record
    records,summary=generate(target=3,seed=31,attempts=100,deadline_ms=100,node_cap=20_000,specs=((6,4,26,28),(10,5,25,27),(15,5,25,27)))
    eligible=0
    for item in records:
        result=validate_record(item); assert result.accepted
        eligible+=int(result.eligible)
        if item["solver"]["status"]=="exact_partial": assert result.eligible
        if item["solver"]["status"]=="unknown": assert not result.eligible
    assert eligible==summary["counts"]["ground_truth_denominator"]
