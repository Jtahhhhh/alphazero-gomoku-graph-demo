"""Solver, flat-proof, and VCF-certificate semantic extraction."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.h1_schema import state_from_record
from azgomoku.proof_replay import replay_flat_proof
from azgomoku.tactics import windows

from .epistemic import EpistemicClass
from .extract_state import (
    ensure_cell_entity,
    ensure_line_entity,
    ensure_move_entity,
    extract_state,
)
from .identity import (
    forced_response_identity,
    make_fact,
    make_provenance,
    normalize_flat_proof,
    proof_identity,
    proof_node_identity,
)
from .predicates import Predicate
from .schema import Entity, EntityType, SemanticArtifact


def _window_relation(state, cell_list: Sequence[int]) -> str:
    target = tuple(sorted(map(int, cell_list)))
    matches = {
        relation
        for relation, known in windows(state.size, state.win_length)
        if tuple(sorted(map(int, known))) == target
    }
    if len(matches) != 1:
        raise ValueError(f"proof window has ambiguous or missing relation: {list(cell_list)}")
    return next(iter(matches))


def _visit_certificate_tree(
    artifact: SemanticArtifact,
    state,
    proof_entity: Entity,
    proof_identity_value,
    node: Mapping[str, Any],
    path: tuple[int, ...],
    provenance_id: str,
) -> None:
    identity = proof_node_identity(state, proof_identity_value, path, node)
    node_entity = artifact.add_entity(
        Entity(
            identity.entity_id,
            EntityType.ProofNode,
            state_identifier(state),
            {
                "path": list(path),
                "player_to_move": int(node["player_to_move"]),
                "move": None if node.get("move") is None else int(node["move"]),
                "node_type": str(node["node_type"]),
                "terminal": node.get("terminal"),
                "child_count": len(node.get("children", [])),
            },
            identity.canonical_key,
        )
    )
    artifact.add_fact(
        make_fact(
            subject_id=proof_entity.entity_id,
            predicate=Predicate.CONTAINS,
            object_id=node_entity.entity_id,
            provenance_id=provenance_id,
            epistemic_class=EpistemicClass.DERIVED,
        )
    )
    children = list(node.get("children", []))
    if node.get("node_type") == "AND":
        for child in children:
            if child.get("move") is None:
                raise ValueError("VCF AND child lacks a forced response move")
            response_identity = forced_response_identity(
                state,
                int(child["move"]),
                scope_entity_id=node_entity.entity_id,
                scope_canonical_key=node_entity.canonical_key,
            )
            response = artifact.add_entity(
                Entity(
                    response_identity.entity_id,
                    EntityType.ForcedResponse,
                    state_identifier(state),
                    {
                        "action": int(child["move"]),
                        "scope_entity_id": node_entity.entity_id,
                        "scope_kind": "ProofNode",
                    },
                    response_identity.canonical_key,
                )
            )
            artifact.add_fact(
                make_fact(
                    subject_id=proof_entity.entity_id,
                    predicate=Predicate.REQUIRES,
                    object_id=response.entity_id,
                    provenance_id=provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )
    for index, child in enumerate(children):
        _visit_certificate_tree(
            artifact,
            state,
            proof_entity,
            proof_identity_value,
            child,
            path + (index,),
            provenance_id,
        )


def extract_record_proofs(
    record: dict,
    artifact: SemanticArtifact | None = None,
    *,
    artifact_ref: str | None = None,
) -> SemanticArtifact:
    state = state_from_record(record)
    artifact = extract_state(state) if artifact is None else artifact
    state_id = state_identifier(state)
    if record.get("state_id") != state_id:
        raise ValueError("record state_id does not match state content")
    solver = record["solver"]
    source_ref = artifact_ref or f"state_id:{state_id}"

    if solver.get("status") == "exact_complete":
        if solver.get("method") != "full_minimax" or not solver.get("optimal_actions_complete"):
            raise ValueError("exact_complete solver record lacks full-minimax completeness")
        exact_source = artifact.add_provenance(
            make_provenance(
                state_id=state_id,
                source_kind="solver",
                source_file="azgomoku/solver.py",
                source_function="solve_actions",
                method="full_minimax",
                status="exact_complete",
                generator_version=record.get("provenance", {}).get("generator_version"),
                artifact_ref=source_ref,
                budget=solver.get("budget"),
            )
        )
        board_id = f"state:{state_id}"
        for action in map(int, solver.get("optimal_actions") or []):
            move = ensure_move_entity(artifact, state, action)
            artifact.add_fact(
                make_fact(
                    subject_id=move.entity_id,
                    predicate=Predicate.OPTIMAL_IN,
                    object_id=board_id,
                    provenance_id=exact_source.provenance_id,
                    epistemic_class=EpistemicClass.EXACT,
                )
            )
        for key, value in (solver.get("action_values") or {}).items():
            move = ensure_move_entity(artifact, state, int(key))
            artifact.add_fact(
                make_fact(
                    subject_id=move.entity_id,
                    predicate=Predicate.HAS_ACTION_VALUE,
                    value=int(value),
                    provenance_id=exact_source.provenance_id,
                    epistemic_class=EpistemicClass.EXACT,
                )
            )

    certificates = {
        item["certificate_id"]: item for item in record.get("proof_certificates", [])
    }
    for proof in record.get("valid_proofs", []):
        certificate = certificates.get(proof.get("certificate_id"))
        if not replay_flat_proof(state, proof, certificate):
            raise ValueError(f"flat proof did not replay for {state_id}: action={proof.get('action')}")
        identity = proof_identity(state, proof)
        normalized = normalize_flat_proof(proof)
        proof_entity = artifact.add_entity(
            Entity(
                identity.entity_id,
                EntityType.Proof,
                state_id,
                {
                    **normalized,
                    "certificate_id": proof.get("certificate_id"),
                },
                identity.canonical_key,
            )
        )
        method = str(proof.get("proof_method"))
        evidence_id = proof.get("certificate_id") or proof_entity.entity_id
        certified_source = artifact.add_provenance(
            make_provenance(
                state_id=state_id,
                source_kind="vcf_certificate" if method == "vcf" else "tactical_replay",
                source_file="investigation/e3b_common.py",
                source_function="replay_flat_proof",
                method=method,
                status="replay_passed",
                generator_version=record.get("proof_annotation", {}).get("version"),
                artifact_ref=source_ref,
                proof_or_certificate_id=evidence_id,
                budget=solver.get("budget"),
            )
        )
        geometry_source = artifact.add_provenance(
            make_provenance(
                state_id=state_id,
                source_kind="proof_geometry",
                source_file="azgomoku/semantic/extract_proofs.py",
                source_function="extract_record_proofs",
                method="deterministic_flat_proof_materialization",
                status="derived_from_replayed_proof",
                generator_version=record.get("proof_annotation", {}).get("version"),
                artifact_ref=source_ref,
                proof_or_certificate_id=evidence_id,
            )
        )
        move = ensure_move_entity(artifact, state, int(proof["action"]))
        artifact.add_fact(
            make_fact(
                subject_id=proof_entity.entity_id,
                predicate=Predicate.SUPPORTS,
                object_id=move.entity_id,
                provenance_id=certified_source.provenance_id,
                epistemic_class=EpistemicClass.CERTIFIED,
            )
        )
        for action in map(int, proof.get("critical_cells", [])):
            cell = ensure_cell_entity(artifact, state, action)
            artifact.add_fact(
                make_fact(
                    subject_id=proof_entity.entity_id,
                    predicate=Predicate.USES_CELL,
                    object_id=cell.entity_id,
                    provenance_id=geometry_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )
        for cell_list in proof.get("windows", []):
            relation = _window_relation(state, cell_list)
            line = ensure_line_entity(artifact, state, relation, cell_list)
            artifact.add_fact(
                make_fact(
                    subject_id=proof_entity.entity_id,
                    predicate=Predicate.HAS_WINDOW,
                    object_id=line.entity_id,
                    provenance_id=geometry_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )
        if certificate is not None:
            _visit_certificate_tree(
                artifact,
                state,
                proof_entity,
                identity,
                certificate["tree"],
                (),
                geometry_source.provenance_id,
            )
    return artifact
