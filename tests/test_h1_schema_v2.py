from copy import deepcopy
from dataclasses import replace

import numpy as np

from azgomoku.game import GomokuState
from azgomoku.ground_truth import GroundTruthBudget,route_ground_truth
from azgomoku.h1_schema import make_record,validate_record
from azgomoku.solver import solve_actions


def complete_record():
    state=GomokuState(np.asarray([[1,-1,1],[-1,1,-1],[-1,1,0]],dtype=np.int8),to_play=1,win_length=3)
    result=route_ground_truth(state,GroundTruthBudget(100_000,2_000),empty_bounds={})
    return make_record(state,[],result,7,generator_version="test",ply=8,dedup_mode="state_id")


def partial_record():
    state=GomokuState(np.asarray([[0]*6,[0]*6,[1,1,1,0,0,0],[0]*6,[0]*6,[0]*6],dtype=np.int8),to_play=1,win_length=4)
    result=route_ground_truth(state,GroundTruthBudget(100_000,2_000),empty_bounds={6:0})
    return make_record(state,[],result,7,generator_version="test",ply=3,dedup_mode="state_id")


def test_v1_record_remains_backward_compatible():
    record=complete_record(); record["schema_version"]=1; record["solver"]=solve_actions(GomokuState(np.asarray(record["state"]["board"],dtype=np.int8),1,record["state"]["last_move"],3)).dict()
    result=validate_record(record)
    assert result.accepted and result.eligible and result.label_kind=="exact_complete"


def test_v2_complete_partial_and_unknown_validate_with_correct_eligibility():
    assert validate_record(complete_record()).eligible
    assert validate_record(partial_record()).eligible
    record=partial_record(); record["solver"].update({"status":"unknown","value":None,"optimal_actions":None,"action_values":None,"proof":None,"valid_proofs":[],"unknown_reason":"exhausted"}); record["valid_proofs"]=[]
    result=validate_record(record)
    assert result.accepted and not result.eligible and result.label_kind=="unknown"


def test_missing_or_unknown_status_is_rejected():
    for value in (None,"invented"):
        record=complete_record()
        if value is None: record["solver"].pop("status")
        else: record["solver"]["status"]=value
        assert not validate_record(record).accepted


def test_missing_completeness_fails_closed_to_partial_treatment():
    record=partial_record(); record["solver"].pop("optimal_actions_complete")
    result=validate_record(record)
    assert result.accepted and result.eligible and result.label_kind=="exact_partial"
    assert result.record["_validation"]["label_kind"]=="exact_partial"
    assert result.record["solver"]["optimal_actions_complete"] is False
    record=complete_record(); record["solver"].pop("optimal_actions_complete")
    result=validate_record(record)
    assert result.accepted and result.eligible and result.label_kind=="exact_partial"


def test_missing_required_field_and_wrong_perspective_are_rejected():
    record=complete_record(); record["solver"].pop("nodes")
    assert not validate_record(record).accepted
    record=complete_record(); record["solver"]["perspective"]["convention_version"]=1
    assert not validate_record(record).accepted


def test_tampered_certificate_missing_and_child_is_rejected_on_read():
    record=partial_record(); first=record["solver"]["proof"]["children"][0]
    if first["children"]: first["children"]=first["children"][:-1]
    else: first["move"]=(first["move"]+1)%36
    assert not validate_record(record).accepted
