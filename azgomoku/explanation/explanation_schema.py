"""Versioned schema construction for state-specific decision evidence."""

import hashlib
import json


SCHEMA_VERSION = "1.0"
EDGE_FILTER = "selected_related_then_candidate_then_global_top_k"


def cell(action, size):
    action = int(action)
    return {"action": action, "row": action // size, "col": action % size}


def state_identifier(state):
    canonical = json.dumps({"board":state.board.astype(int).tolist(),"current_player":int(state.to_play),"last_move":int(state.last_move),"win_length":int(state.win_length)},sort_keys=True,separators=(",",":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def make_document(state, model_type, checkpoint, selected_move, top_k_edges):
    last = None if state.last_move < 0 else [state.last_move // state.size, state.last_move % state.size]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "state_specific_decision_evidence",
        "model": {"type": model_type, "checkpoint": str(checkpoint) if checkpoint else None},
        "state_id": state_identifier(state),
        "state": {
            "board_size": state.size, "win_length": state.win_length,
            "current_player": int(state.to_play), "last_move": last,
            "board": state.board.astype(int).tolist(),
            "legal_actions": [int(a) for a in state.legal_actions()],
        },
        "selected_move": cell(selected_move, state.size),
        "network": {}, "mcts": {},
        "graph_evidence": {"attention_available": False, "edges": []},
        "semantic_attention": {},
        "rendering": {"top_k_edges": int(top_k_edges), "edge_filter": EDGE_FILTER, "head_aggregation": "mean across attention heads"},
        "runtime_ms": {},
        "limitations": [
            "Attention weights are state-specific model evidence, not causal explanations.",
            "MCTS decision trace and neural-network evidence are reported separately.",
        ],
    }
