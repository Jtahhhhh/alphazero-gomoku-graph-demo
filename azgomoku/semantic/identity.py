"""Stable raw identities and D4-canonical semantic equivalence keys."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.symmetry import (
    transform_action,
    transform_flat_proof,
    transform_relation,
    transform_state,
)

from .epistemic import EpistemicClass
from .predicates import Predicate
from .schema import Provenance, RelationFact


_PAIR_CANONICAL_CACHE: dict[str, str] = {}


@dataclass(frozen=True)
class SemanticIdentity:
    entity_id: str
    canonical_key: str


def _jsonable(value: Any) -> Any:
    if hasattr(value, "value") and isinstance(getattr(value, "value"), str):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_digest(value: Any, length: int = 24) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _state_payload(state) -> dict[str, Any]:
    return {
        "board": state.board.astype(int).tolist(),
        "current_player": int(state.to_play),
        "last_move": int(state.last_move),
        "win_length": int(state.win_length),
    }


def _pair_canonical_key(
    state,
    kind: str,
    transform_payload: Callable[[int], Any],
) -> str:
    def candidate(symmetry: int) -> str:
        return canonical_json(
            {
                "state": _state_payload(transform_state(state, symmetry)),
                "semantic_type": kind,
                "payload": transform_payload(symmetry),
            }
        )

    raw = candidate(0)
    cached = _PAIR_CANONICAL_CACHE.get(raw)
    if cached is not None:
        return cached
    candidates = [raw, *(candidate(symmetry) for symmetry in range(1, 8))]
    canonical = f"d4:{kind}:{hashlib.sha256(min(candidates).encode('utf-8')).hexdigest()[:24]}"
    # Every member of an orbit is a valid cache key. Full D4 gates then pay the
    # canonicalization cost once per semantic object, not once per orientation.
    for item in candidates:
        _PAIR_CANONICAL_CACHE[item] = canonical
    return canonical


def d4_canonical_state_key(state) -> str:
    return _pair_canonical_key(state, "BoardState", lambda _symmetry: {})


def board_state_identity(state) -> SemanticIdentity:
    raw = state_identifier(state)
    return SemanticIdentity(f"state:{raw}", d4_canonical_state_key(state))


def cell_identity(state, action: int) -> SemanticIdentity:
    action = int(action)
    row, col = divmod(action, state.size)
    if action < 0 or action >= state.size * state.size:
        raise ValueError("cell action is outside the board")
    raw_state = state_identifier(state)
    return SemanticIdentity(
        f"cell:{raw_state}:r{row}c{col}",
        _pair_canonical_key(
            state,
            "Cell",
            lambda symmetry: {"action": transform_action(action, state.size, symmetry)},
        ),
    )


def move_identity(state, action: int) -> SemanticIdentity:
    action = int(action)
    if action < 0 or action >= state.size * state.size:
        raise ValueError("move action is outside the board")
    raw_state = state_identifier(state)
    return SemanticIdentity(
        f"move:{raw_state}:a{action}",
        _pair_canonical_key(
            state,
            "Move",
            lambda symmetry: {"action": transform_action(action, state.size, symmetry)},
        ),
    )


def _normalized_line_cells(cells: Sequence[int]) -> tuple[int, ...]:
    ordered = tuple(map(int, cells))
    reversed_cells = tuple(reversed(ordered))
    return min(ordered, reversed_cells)


def line_window_identity(
    state, relation: str, cells: Sequence[int], player: int | None = None
) -> SemanticIdentity:
    normalized = _normalized_line_cells(cells)
    content = {"relation": str(relation), "cells": normalized, "player": player}
    raw_state = state_identifier(state)

    def transformed(symmetry: int) -> dict[str, Any]:
        mapped = [transform_action(cell, state.size, symmetry) for cell in normalized]
        return {
            "relation": transform_relation(str(relation), symmetry),
            "cells": _normalized_line_cells(mapped),
            "player": player,
        }

    return SemanticIdentity(
        f"line:{raw_state}:{stable_digest(content)}",
        _pair_canonical_key(state, "LineWindow", transformed),
    )


def winning_threat_identity(
    state,
    player: int,
    relation: str,
    window: Sequence[int],
    completion: int,
) -> SemanticIdentity:
    normalized = _normalized_line_cells(window)
    content = {
        "player": int(player),
        "relation": str(relation),
        "window": normalized,
        "completion": int(completion),
    }
    raw_state = state_identifier(state)

    def transformed(symmetry: int) -> dict[str, Any]:
        mapped = [transform_action(cell, state.size, symmetry) for cell in normalized]
        return {
            "player": int(player),
            "relation": transform_relation(str(relation), symmetry),
            "window": _normalized_line_cells(mapped),
            "completion": transform_action(int(completion), state.size, symmetry),
        }

    return SemanticIdentity(
        f"threat:{raw_state}:{stable_digest(content)}",
        _pair_canonical_key(state, "WinningThreat", transformed),
    )


def defense_set_identity(
    state,
    attacker: int,
    completions: Sequence[int],
    blocking_moves: Sequence[int],
    unstoppable: bool,
) -> SemanticIdentity:
    content = {
        "attacker": int(attacker),
        "completions": sorted(map(int, completions)),
        "blocking_moves": sorted(map(int, blocking_moves)),
        "unstoppable": bool(unstoppable),
    }
    raw_state = state_identifier(state)

    def transformed(symmetry: int) -> dict[str, Any]:
        return {
            "attacker": int(attacker),
            "completions": sorted(
                transform_action(int(action), state.size, symmetry) for action in completions
            ),
            "blocking_moves": sorted(
                transform_action(int(action), state.size, symmetry) for action in blocking_moves
            ),
            "unstoppable": bool(unstoppable),
        }

    return SemanticIdentity(
        f"defense_set:{raw_state}:{stable_digest(content)}",
        _pair_canonical_key(state, "DefenseSet", transformed),
    )


def forced_response_identity(
    state,
    action: int,
    *,
    scope_entity_id: str,
    scope_canonical_key: str,
) -> SemanticIdentity:
    action = int(action)
    return SemanticIdentity(
        f"forced_response:{scope_entity_id}:a{action}",
        _pair_canonical_key(
            state,
            "ForcedResponse",
            lambda symmetry: {
                "scope": scope_canonical_key,
                "action": transform_action(action, state.size, symmetry),
            },
        ),
    )


def normalize_flat_proof(proof: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": int(proof["action"]),
        "concepts": sorted(map(str, proof.get("concepts", []))),
        "critical_cells": sorted(map(int, proof.get("critical_cells", []))),
        "critical_relations": sorted(map(str, proof.get("critical_relations", []))),
        "windows": sorted(
            tuple(sorted(map(int, window))) for window in proof.get("windows", [])
        ),
        "proof_method": proof.get("proof_method"),
        "proof_status": proof.get("proof_status"),
    }


def proof_identity(state, proof: Mapping[str, Any]) -> SemanticIdentity:
    normalized = normalize_flat_proof(proof)
    raw_state = state_identifier(state)

    def transformed(symmetry: int) -> dict[str, Any]:
        return normalize_flat_proof(transform_flat_proof(dict(proof), state.size, symmetry))

    return SemanticIdentity(
        f"proof:{raw_state}:{stable_digest(normalized)}",
        _pair_canonical_key(state, "Proof", transformed),
    )


def transform_proof_node_dict(node: Mapping[str, Any], size: int, symmetry: int) -> dict[str, Any]:
    children = [transform_proof_node_dict(child, size, symmetry) for child in node.get("children", [])]
    children.sort(key=canonical_json)
    move = node.get("move")
    return {
        "player_to_move": int(node["player_to_move"]),
        "move": None if move is None else transform_action(int(move), size, symmetry),
        "node_type": str(node["node_type"]),
        "children": children,
        "terminal": node.get("terminal"),
    }


def proof_node_identity(
    state,
    proof: SemanticIdentity,
    path: Sequence[int],
    node: Mapping[str, Any],
) -> SemanticIdentity:
    path_text = "root" if not path else "/".join(map(str, path))
    return SemanticIdentity(
        f"proofnode:{proof.entity_id}:{path_text}",
        _pair_canonical_key(
            state,
            "ProofNode",
            lambda symmetry: {
                "proof": proof.canonical_key,
                "subtree": transform_proof_node_dict(node, state.size, symmetry),
            },
        ),
    )


def structural_edge_identity(
    state, relation: str, source: int, target: int
) -> SemanticIdentity:
    relation = str(relation)
    source, target = int(source), int(target)
    raw_state = state_identifier(state)
    return SemanticIdentity(
        f"edge:{raw_state}:{relation}:{source}:{target}",
        _pair_canonical_key(
            state,
            "StructuralEdge",
            lambda symmetry: {
                "relation": transform_relation(relation, symmetry),
                "source": transform_action(source, state.size, symmetry),
                "target": transform_action(target, state.size, symmetry),
            },
        ),
    )


def attention_observation_identity(
    state,
    checkpoint_sha: str,
    relation: str,
    source: int,
    target: int,
    layer: str,
) -> SemanticIdentity:
    edge = structural_edge_identity(state, relation, source, target)
    legacy = f"{relation}:{int(source)}:{int(target)}"
    return SemanticIdentity(
        f"attention:{state_identifier(state)}:{checkpoint_sha}:{legacy}:{layer}",
        f"d4:AttentionObservation:{stable_digest({'checkpoint_sha': str(checkpoint_sha), 'layer': str(layer), 'edge': edge.canonical_key})}",
    )


def search_config_hash(search_config: Mapping[str, Any]) -> str:
    return stable_digest(dict(search_config))


def mcts_candidate_identity(
    state, search_config: Mapping[str, Any], action: int
) -> SemanticIdentity:
    config_hash = search_config_hash(search_config)
    action = int(action)
    move = move_identity(state, action)
    return SemanticIdentity(
        f"mcts:{state_identifier(state)}:{config_hash}:a{action}",
        f"d4:MCTSCandidate:{stable_digest({'search_config_hash': config_hash, 'move': move.canonical_key})}",
    )


def make_provenance(**fields: Any) -> Provenance:
    payload = dict(fields)
    payload.pop("provenance_id", None)
    return Provenance(provenance_id=f"prov:{stable_digest(payload)}", **payload)


def make_fact(
    *,
    subject_id: str,
    predicate: Predicate | str,
    provenance_id: str,
    epistemic_class: EpistemicClass | str,
    object_id: str | None = None,
    value: Any | None = None,
) -> RelationFact:
    payload = {
        "subject_id": subject_id,
        "predicate": predicate,
        "object_id": object_id,
        "value": value,
        "provenance_id": provenance_id,
        "epistemic_class": epistemic_class,
    }
    return RelationFact(fact_id=f"fact:{stable_digest(payload)}", **payload)
