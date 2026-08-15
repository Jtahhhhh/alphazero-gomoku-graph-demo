"""Deterministic BoardState, Cell, Move, LineWindow, and edge extraction."""

from __future__ import annotations

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.graph import cell_edge_records
from azgomoku.tactics import windows

from .epistemic import EpistemicClass
from .identity import (
    board_state_identity,
    cell_identity,
    line_window_identity,
    make_fact,
    make_provenance,
    move_identity,
    structural_edge_identity,
)
from .predicates import Predicate
from .schema import Entity, EntityType, SemanticArtifact


def ensure_cell_entity(artifact: SemanticArtifact, state, action: int) -> Entity:
    action = int(action)
    identity = cell_identity(state, action)
    row, col = divmod(action, state.size)
    return artifact.add_entity(
        Entity(
            identity.entity_id,
            EntityType.Cell,
            state_identifier(state),
            {
                "action": action,
                "row": row,
                "col": col,
                "occupancy": int(state.board[row, col]),
            },
            identity.canonical_key,
        )
    )


def ensure_move_entity(artifact: SemanticArtifact, state, action: int, **attributes) -> Entity:
    action = int(action)
    identity = move_identity(state, action)
    row, col = divmod(action, state.size)
    payload = {
        "action": action,
        "row": row,
        "col": col,
        "player": int(state.to_play),
        **attributes,
    }
    return artifact.add_entity(
        Entity(
            identity.entity_id,
            EntityType.Move,
            state_identifier(state),
            payload,
            identity.canonical_key,
        )
    )


def ensure_line_entity(
    artifact: SemanticArtifact,
    state,
    relation: str,
    cells,
    *,
    player: int | None = None,
) -> Entity:
    identity = line_window_identity(state, relation, cells, player)
    return artifact.add_entity(
        Entity(
            identity.entity_id,
            EntityType.LineWindow,
            state_identifier(state),
            {
                "relation": str(relation),
                "cells": list(map(int, cells)),
                "player": player,
            },
            identity.canonical_key,
        )
    )


def extract_state(state) -> SemanticArtifact:
    artifact = SemanticArtifact()
    state_id = state_identifier(state)
    board_identity = board_state_identity(state)
    board_entity = artifact.add_entity(
        Entity(
            board_identity.entity_id,
            EntityType.BoardState,
            state_id,
            {
                "board_size": int(state.size),
                "win_length": int(state.win_length),
                "current_player": int(state.to_play),
                "last_move": int(state.last_move),
                "board": state.board.astype(int).tolist(),
            },
            board_identity.canonical_key,
        )
    )

    state_source = artifact.add_provenance(
        make_provenance(
            state_id=state_id,
            source_kind="state_structure",
            source_file="azgomoku/game.py",
            source_function="GomokuState",
            method="deterministic_materialization",
            status="derived",
        )
    )
    line_source = artifact.add_provenance(
        make_provenance(
            state_id=state_id,
            source_kind="line_geometry",
            source_file="azgomoku/tactics.py",
            source_function="windows",
            method="deterministic_window_enumeration",
            status="derived",
        )
    )
    edge_source = artifact.add_provenance(
        make_provenance(
            state_id=state_id,
            source_kind="structural_graph",
            source_file="azgomoku/graph.py",
            source_function="cell_edge_records",
            method="deterministic_adjacency",
            status="derived",
        )
    )

    for action in range(state.size * state.size):
        cell = ensure_cell_entity(artifact, state, action)
        artifact.add_fact(
            make_fact(
                subject_id=board_entity.entity_id,
                predicate=Predicate.CONTAINS,
                object_id=cell.entity_id,
                provenance_id=state_source.provenance_id,
                epistemic_class=EpistemicClass.DERIVED,
            )
        )

    for action in map(int, state.legal_actions()):
        move = ensure_move_entity(artifact, state, action)
        cell = ensure_cell_entity(artifact, state, action)
        artifact.add_fact(
            make_fact(
                subject_id=move.entity_id,
                predicate=Predicate.PLAYED_AT,
                object_id=cell.entity_id,
                provenance_id=state_source.provenance_id,
                epistemic_class=EpistemicClass.DERIVED,
            )
        )

    for relation, cells in windows(state.size, state.win_length):
        line = ensure_line_entity(artifact, state, relation, cells)
        for action in cells:
            cell = ensure_cell_entity(artifact, state, action)
            artifact.add_fact(
                make_fact(
                    subject_id=line.entity_id,
                    predicate=Predicate.USES_CELL,
                    object_id=cell.entity_id,
                    provenance_id=line_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )

    for record in cell_edge_records(state.size):
        identity = structural_edge_identity(
            state, record["relation"], record["source"], record["target"]
        )
        edge = artifact.add_entity(
            Entity(
                identity.entity_id,
                EntityType.StructuralEdge,
                state_id,
                {
                    "legacy_edge_id": record["edge_id"],
                    "relation": record["relation"],
                    "source_action": int(record["source"]),
                    "target_action": int(record["target"]),
                    "directed": True,
                },
                identity.canonical_key,
            )
        )
        for endpoint in (record["source"], record["target"]):
            cell = ensure_cell_entity(artifact, state, endpoint)
            artifact.add_fact(
                make_fact(
                    subject_id=edge.entity_id,
                    predicate=Predicate.CONNECTS,
                    object_id=cell.entity_id,
                    provenance_id=edge_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )
    return artifact
