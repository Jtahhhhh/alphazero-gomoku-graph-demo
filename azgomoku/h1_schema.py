"""H1 schema-v2 writer and fail-closed v1/v2 reader."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .explanation.explanation_schema import state_identifier
from .game import GomokuState
from .vcf import ProofNode,replay_vcf_proof


STATUSES={"exact_complete","exact_partial","unknown"}
METHODS={"full_minimax","vcf"}
UNKNOWN_REASONS={"exhausted","budget",None}
REQUIRED_SOLVER_FIELDS={
    "status","method","value","optimal_actions","optimal_actions_complete","action_values",
    "proof","nodes","elapsed_ms","budget","unknown_reason","coverage_note","perspective",
}


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    eligible: bool
    label_kind: str | None
    record: dict | None
    errors: tuple[str,...]


def state_from_record(record: dict) -> GomokuState:
    item=record["state"]
    board=np.asarray(item["board"],dtype=np.int8)
    if board.ndim!=2 or board.shape[0]!=board.shape[1] or board.shape[0]!=int(item["board_size"]):
        raise ValueError("invalid board shape")
    return GomokuState(board,int(item["current_player"]),int(item["last_move"]),int(item["win_length"]))


def proof_node_from_dict(item: dict) -> ProofNode:
    if not isinstance(item,dict): raise ValueError("proof must be an object")
    return ProofNode(
        player_to_move=int(item["player_to_move"]),
        move=None if item["move"] is None else int(item["move"]),
        node_type=item["node_type"],
        children=tuple(proof_node_from_dict(child) for child in item.get("children",[])),
        terminal=item.get("terminal"),
    )


def make_record(state,history,result,seed,*,generator_version,ply,dedup_mode,provenance_extra=None):
    return {
        "schema_version":2,
        "state_id":state_identifier(state),
        "state":{
            "board_size":state.size,"win_length":state.win_length,"current_player":int(state.to_play),
            "last_move":int(state.last_move),"board":state.board.astype(int).tolist(),
            "legal_actions":[int(action) for action in state.legal_actions()],
        },
        "provenance":{
            "generator_version":generator_version,"seed":int(seed),"board_size":state.size,
            "ply":int(ply),"empty_count":len(state.legal_actions()),"history":[int(action) for action in history],
            "dedup_mode":dedup_mode,**(provenance_extra or {}),
        },
        "solver":result.dict(),
        "valid_proofs":list(result.valid_proofs),
    }


def _reject(errors,*items):
    return ValidationResult(False,False,None,None,tuple(errors)+tuple(items))


def validate_record(record: dict) -> ValidationResult:
    errors=[]
    if not isinstance(record,dict): return _reject(errors,"record is not an object")
    version=record.get("schema_version")
    if version==1:
        try:
            state=state_from_record(record); solver=dict(record["solver"])
            if record.get("state_id")!=state_identifier(state): raise ValueError("state_id mismatch")
            status=solver.get("status")
            if status=="exact":
                solver.update({"status":"exact_complete","method":"full_minimax","optimal_actions_complete":True,"legacy_missing":True})
                normalized={**record,"solver":solver}
                return ValidationResult(True,True,"exact_complete",normalized,())
            if status in ("timeout","node_budget"):
                solver.update({"status":"unknown","method":"full_minimax","optimal_actions_complete":False,"unknown_reason":"budget","legacy_missing":True})
                return ValidationResult(True,False,"unknown",{**record,"solver":solver},())
            return _reject(errors,"unsupported v1 solver status")
        except (KeyError,TypeError,ValueError) as exc: return _reject(errors,f"invalid v1 record: {exc}")
    if version!=2: return _reject(errors,"unsupported or missing schema_version")
    try:
        state=state_from_record(record)
        if record.get("state_id")!=state_identifier(state): errors.append("state_id mismatch")
        solver=record.get("solver")
        if not isinstance(solver,dict): return _reject(errors,"missing solver")
        missing=REQUIRED_SOLVER_FIELDS-set(solver)
        completeness_missing="optimal_actions_complete" in missing
        # Missing completeness is the one explicitly permitted fail-closed default.
        missing_without_completeness=missing-{"optimal_actions_complete"}
        if missing_without_completeness: errors.append("missing solver fields: "+",".join(sorted(missing_without_completeness)))
        status=solver.get("status")
        if status not in STATUSES: errors.append("invalid or missing status")
        method=solver.get("method")
        if method not in METHODS: errors.append("invalid method")
        complete=bool(solver.get("optimal_actions_complete",False))
        perspective=solver.get("perspective")
        if not isinstance(perspective,dict) or perspective.get("convention_version")!=2 or perspective.get("value")!="player_to_move_at_state": errors.append("invalid perspective convention")
        budget=solver.get("budget")
        if not isinstance(budget,dict) or not {"node_cap","time_cap_ms"}<=set(budget): errors.append("invalid budget")
        legal=set(map(int,state.legal_actions()))
        actions=solver.get("optimal_actions")
        action_values=solver.get("action_values")
        parsed_actions=None if actions is None else tuple(map(int,actions))
        parsed_values=None if action_values is None else {int(key):int(value) for key,value in action_values.items()}
        if parsed_actions is not None and not set(parsed_actions)<=legal: errors.append("illegal optimal action")
        if status=="exact_complete":
            if method!="full_minimax" or (not complete and not completeness_missing) or solver.get("value") is None: errors.append("invalid exact_complete semantics")
            if parsed_values is None or set(parsed_values)!=legal: errors.append("incomplete exact action map")
            if solver.get("proof") is not None or solver.get("unknown_reason") is not None: errors.append("invalid exact_complete proof/reason")
        elif status=="exact_partial":
            if method!="vcf" or complete or solver.get("value")!=1: errors.append("invalid exact_partial semantics")
            if not parsed_actions or parsed_values is None or any(parsed_values.get(action)!=1 for action in parsed_actions): errors.append("invalid proven actions")
            try:
                proof=proof_node_from_dict(solver.get("proof"))
                if not replay_vcf_proof(state,proof): errors.append("proof replay failed")
            except (KeyError,TypeError,ValueError): errors.append("proof replay failed")
            if solver.get("unknown_reason") is not None: errors.append("partial carries unknown_reason")
        elif status=="unknown":
            if any(solver.get(key) is not None for key in ("value","optimal_actions","action_values","proof")) or complete: errors.append("unknown carries a label")
            if solver.get("unknown_reason") not in ("budget","exhausted"): errors.append("invalid unknown_reason")
            if record.get("valid_proofs") not in ([],()): errors.append("unknown carries valid proofs")
        if solver.get("unknown_reason") not in UNKNOWN_REASONS: errors.append("invalid unknown_reason")
        valid_proofs=record.get("valid_proofs")
        if not isinstance(valid_proofs,list): errors.append("invalid valid_proofs")
        elif status=="exact_partial":
            if not valid_proofs: errors.append("partial lacks reduced proofs")
            for flat in valid_proofs:
                if not isinstance(flat,dict) or flat.get("proof_method")!="vcf" or flat.get("proof_status")!="exact": errors.append("invalid reduced proof")
                elif int(flat.get("action",-1)) not in set(parsed_actions or ()): errors.append("reduced proof action is not proven")
        provenance=record.get("provenance")
        required_provenance={"generator_version","seed","board_size","ply","empty_count"}
        if not isinstance(provenance,dict) or not required_provenance<=set(provenance): errors.append("invalid provenance")
        if errors: return _reject(errors)
        normalized_solver={**solver,"optimal_actions_complete":complete}
        treatment="exact_partial" if completeness_missing and status=="exact_complete" else status
        normalized={**record,"solver":normalized_solver,"_validation":{"eligible":status!="unknown","label_kind":treatment}}
        return ValidationResult(True,status!="unknown",treatment,normalized,())
    except (KeyError,TypeError,ValueError) as exc:
        return _reject(errors,f"invalid v2 record: {exc}")


def read_records(records) -> tuple[list[dict],list[dict]]:
    accepted=[]; rejected=[]
    for index,record in enumerate(records):
        result=validate_record(record)
        if result.accepted: accepted.append(result.record)
        else: rejected.append({"index":index,"errors":list(result.errors)})
    return accepted,rejected
