"""Thin semantic adapter over existing deterministic tactical primitives."""

from __future__ import annotations

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.tactics import mandatory_defenses, threat_moves, winning_completions

from .epistemic import EpistemicClass
from .extract_state import ensure_cell_entity, ensure_move_entity, extract_state
from .identity import (
    defense_set_identity,
    forced_response_identity,
    make_fact,
    make_provenance,
    winning_threat_identity,
)
from .predicates import Predicate
from .schema import Entity, EntityType, SemanticArtifact


def _add_threat(
    artifact: SemanticArtifact,
    state,
    player: int,
    threat,
    provenance_id: str,
) -> Entity:
    identity = winning_threat_identity(
        state, player, threat.relation, threat.window, threat.completion
    )
    entity = artifact.add_entity(
        Entity(
            identity.entity_id,
            EntityType.WinningThreat,
            state_identifier(state),
            {
                "player": int(player),
                "relation": threat.relation,
                "window": list(map(int, threat.window)),
                "completion": int(threat.completion),
                "semantic_scope": "winning_completion_window",
            },
            identity.canonical_key,
        )
    )
    for action in threat.window:
        cell = ensure_cell_entity(artifact, state, action)
        artifact.add_fact(
            make_fact(
                subject_id=entity.entity_id,
                predicate=Predicate.USES_CELL,
                object_id=cell.entity_id,
                provenance_id=provenance_id,
                epistemic_class=EpistemicClass.DERIVED,
            )
        )
    completion = ensure_cell_entity(artifact, state, threat.completion)
    artifact.add_fact(
        make_fact(
            subject_id=entity.entity_id,
            predicate=Predicate.HAS_COMPLETION,
            object_id=completion.entity_id,
            provenance_id=provenance_id,
            epistemic_class=EpistemicClass.DERIVED,
        )
    )
    artifact.add_fact(
        make_fact(
            subject_id=entity.entity_id,
            predicate=Predicate.HAS_DIRECTION,
            value=threat.relation,
            provenance_id=provenance_id,
            epistemic_class=EpistemicClass.DERIVED,
        )
    )
    return entity


def extract_tactics(state, artifact: SemanticArtifact | None = None) -> SemanticArtifact:
    artifact = extract_state(state) if artifact is None else artifact
    state_id = state_identifier(state)
    threat_source = artifact.add_provenance(
        make_provenance(
            state_id=state_id,
            source_kind="tactical_derivation",
            source_file="azgomoku/tactics.py",
            source_function="winning_completions",
            method="geometry_replay",
            status="derived",
        )
    )
    move_source = artifact.add_provenance(
        make_provenance(
            state_id=state_id,
            source_kind="tactical_derivation",
            source_file="azgomoku/tactics.py",
            source_function="threat_moves",
            method="vcf_candidate_classification",
            status="derived",
        )
    )
    defense_source = artifact.add_provenance(
        make_provenance(
            state_id=state_id,
            source_kind="tactical_derivation",
            source_file="azgomoku/tactics.py",
            source_function="mandatory_defenses",
            method="immediate_threat_defense",
            status="derived",
        )
    )

    immediate_by_player: dict[int, list[tuple[object, Entity]]] = {}
    for player in (-1, 1):
        immediate_by_player[player] = []
        for threat in winning_completions(state, player):
            entity = _add_threat(
                artifact,
                state,
                player,
                threat,
                threat_source.provenance_id,
            )
            immediate_by_player[player].append((threat, entity))

    attacker = -int(state.to_play)
    defense = mandatory_defenses(state, attacker)
    defense_identity = defense_set_identity(
        state,
        attacker,
        defense.completions,
        defense.blocking_moves,
        defense.unstoppable,
    )
    artifact.add_entity(
        Entity(
            defense_identity.entity_id,
            EntityType.DefenseSet,
            state_id,
            {
                "attacker": attacker,
                "defender": int(state.to_play),
                "completions": list(map(int, defense.completions)),
                "blocking_moves": list(map(int, defense.blocking_moves)),
                "unstoppable": bool(defense.unstoppable),
                "scope": "immediate_threats_only",
            },
            defense_identity.canonical_key,
        )
    )
    for threat, threat_entity in immediate_by_player[attacker]:
        for action in defense.blocking_moves:
            if int(action) != int(threat.completion):
                continue
            response_identity = forced_response_identity(
                state,
                action,
                scope_entity_id=threat_entity.entity_id,
                scope_canonical_key=threat_entity.canonical_key,
            )
            response = artifact.add_entity(
                Entity(
                    response_identity.entity_id,
                    EntityType.ForcedResponse,
                    state_id,
                    {
                        "action": int(action),
                        "player": int(state.to_play),
                        "scope_entity_id": threat_entity.entity_id,
                        "scope_kind": "WinningThreat",
                    },
                    response_identity.canonical_key,
                )
            )
            block_move = ensure_move_entity(artifact, state, action)
            artifact.add_fact(
                make_fact(
                    subject_id=threat_entity.entity_id,
                    predicate=Predicate.FORCES,
                    object_id=response.entity_id,
                    provenance_id=defense_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )
            artifact.add_fact(
                make_fact(
                    subject_id=block_move.entity_id,
                    predicate=Predicate.BLOCKS,
                    object_id=threat_entity.entity_id,
                    provenance_id=defense_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )

    for classified in threat_moves(state, int(state.to_play), "vcf"):
        move = ensure_move_entity(
            artifact,
            state,
            classified.move,
            creates_five=bool(classified.creates_five),
            creates_four=bool(classified.creates_four),
            creates_double_four=bool(classified.creates_double_four),
            tactical_scope="existing_vcf_candidate_classifier",
        )
        for threat in classified.fours:
            threat_entity = _add_threat(
                artifact,
                state,
                int(state.to_play),
                threat,
                move_source.provenance_id,
            )
            artifact.add_fact(
                make_fact(
                    subject_id=move.entity_id,
                    predicate=Predicate.CREATES,
                    object_id=threat_entity.entity_id,
                    provenance_id=move_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )
    return artifact
