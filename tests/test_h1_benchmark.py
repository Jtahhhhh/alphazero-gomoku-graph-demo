import json
from pathlib import Path

import numpy as np

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.game import GomokuState
from investigation.generate_h1_benchmark import canonical_key,replay
from models.rgcn import RGCN
from models.rgat import RGAT


PATH=Path("diagnostic/h1_tactical.jsonl")


def records(): return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines()]


def state(record):
    item=record["state"]
    return GomokuState(np.asarray(item["board"],dtype=np.int8),item["current_player"],item["last_move"],item["win_length"])


def test_benchmark_is_exact_legal_deterministic_and_deduplicated():
    items=records(); assert len(items)>=24; keys=set()
    for item in items:
        s=state(item); rebuilt=replay(item["provenance"]["history"],s.size,s.win_length)
        assert np.array_equal(rebuilt.board,s.board) and rebuilt.to_play==s.to_play and rebuilt.last_move==s.last_move
        assert item["solver"]["status"]=="exact" and item["state_id"]==state_identifier(s)
        key=canonical_key(s); assert key not in keys; keys.add(key)
        legal=set(map(int,s.legal_actions())); optimal=set(item["solver"]["optimal_actions"])
        assert optimal<=legal and set(map(int,item["solver"]["action_values"]))==legal
        assert item["valid_proofs"] and all(proof["action"] in legal and proof["action"] in optimal for proof in item["valid_proofs"])


def test_evidence_edges_have_stable_ids_matching_proof_coordinates():
    item=records()[0]; s=state(item); action=item["solver"]["optimal_actions"][0]
    for model in (RGAT(hidden_dim=8,attention_heads=2),RGCN(hidden_dim=8)):
        evidence=collect_model_evidence(s,model,action); edges=evidence["graph_evidence"]["edges"]
        assert len({edge["edge_id"] for edge in edges})==len(edges)
        assert all(edge["source"]["action"]==edge["source"]["row"]*s.size+edge["source"]["col"] for edge in edges)
        relations={relation for proof in item["valid_proofs"] for relation in proof["critical_relations"]}
        assert relations & {edge["relation"] for edge in edges}
