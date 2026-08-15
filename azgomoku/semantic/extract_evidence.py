"""Learned attention and root-MCTS evidence extraction."""

from __future__ import annotations

from typing import Any, Mapping

from azgomoku.explanation.explanation_schema import state_identifier
from azgomoku.metrics.semantic_alignment import critical_ids

from .epistemic import EpistemicClass
from .extract_state import extract_state
from .identity import (
    attention_observation_identity,
    make_fact,
    make_provenance,
    mcts_candidate_identity,
    proof_identity,
)
from .predicates import Predicate
from .schema import Entity, EntityType, SemanticArtifact


def extract_evidence(
    state,
    *,
    model_evidence: Mapping[str, Any] | None = None,
    mcts_trace: Mapping[str, Any] | None = None,
    checkpoint: str,
    checkpoint_sha: str,
    search_config: Mapping[str, Any] | None = None,
    proofs: list[dict] | None = None,
    artifact: SemanticArtifact | None = None,
) -> SemanticArtifact:
    artifact = extract_state(state) if artifact is None else artifact
    state_id = state_identifier(state)
    proofs = proofs or []

    edges = [] if model_evidence is None else list(model_evidence.get("graph_evidence", {}).get("edges", []))
    attention_source = None
    overlap_source = None
    if any(edge.get("attention") is not None for edge in edges):
        attention_source = artifact.add_provenance(
            make_provenance(
                state_id=state_id,
                source_kind="model_evidence",
                source_file="azgomoku/explanation/model_evidence.py",
                source_function="collect_model_evidence",
                method="rgat_attention",
                status="observed",
                artifact_ref=f"state_id:{state_id}",
                model_checkpoint=checkpoint,
            )
        )
        overlap_source = artifact.add_provenance(
            make_provenance(
                state_id=state_id,
                source_kind="attention_proof_overlap",
                source_file="investigation/evaluate_h1.py",
                source_function="critical_ids",
                method="deterministic_edge_window_intersection",
                status="derived",
                artifact_ref=f"state_id:{state_id}",
                model_checkpoint=checkpoint,
            )
        )
    proof_critical = {
        proof_identity(state, proof).entity_id: critical_ids(edges, proof)
        for proof in proofs
    } if edges else {}
    for edge in edges:
        if edge.get("attention") is None:
            continue
        source = int(edge["source"]["action"])
        target = int(edge["target"]["action"])
        layer = str(edge.get("layer", "final"))
        identity = attention_observation_identity(
            state,
            checkpoint_sha,
            edge["relation"],
            source,
            target,
            layer,
        )
        observation = artifact.add_entity(
            Entity(
                identity.entity_id,
                EntityType.AttentionObservation,
                state_id,
                {
                    "legacy_edge_id": edge["edge_id"],
                    "relation": edge["relation"],
                    "source_action": source,
                    "target_action": target,
                    "layer": layer,
                    "head_attention": edge.get("head_attention"),
                    "aggregation": edge.get("attention_aggregation"),
                    "checkpoint_sha": checkpoint_sha,
                },
                identity.canonical_key,
            )
        )
        artifact.add_fact(
            make_fact(
                subject_id=observation.entity_id,
                predicate=Predicate.HAS_WEIGHT,
                value=float(edge["attention"]),
                provenance_id=attention_source.provenance_id,
                epistemic_class=EpistemicClass.LEARNED,
            )
        )
        for proof_id, critical_edge_ids in proof_critical.items():
            if edge["edge_id"] not in critical_edge_ids:
                continue
            if proof_id not in artifact.entities:
                raise ValueError(f"proof entity must be extracted before overlap: {proof_id}")
            artifact.add_fact(
                make_fact(
                    subject_id=observation.entity_id,
                    predicate=Predicate.OVERLAPS,
                    object_id=proof_id,
                    provenance_id=overlap_source.provenance_id,
                    epistemic_class=EpistemicClass.DERIVED,
                )
            )

    if mcts_trace and mcts_trace.get("available"):
        config = dict(search_config or {})
        config.setdefault("playouts", mcts_trace.get("playouts"))
        mcts_source = artifact.add_provenance(
            make_provenance(
                state_id=state_id,
                source_kind="mcts",
                source_file="azgomoku/explanation/mcts_trace.py",
                source_function="extract_mcts_trace",
                method="root_mcts_trace",
                status="observed",
                artifact_ref=f"state_id:{state_id}",
                model_checkpoint=checkpoint,
                budget=config,
            )
        )
        candidates = mcts_trace.get("candidates") or mcts_trace.get("top_candidates", [])
        for item in candidates:
            action = int(item["action"])
            identity = mcts_candidate_identity(state, config, action)
            candidate = artifact.add_entity(
                Entity(
                    identity.entity_id,
                    EntityType.MCTSCandidate,
                    state_id,
                    {
                        "action": action,
                        "row": int(item.get("row", action // state.size)),
                        "col": int(item.get("col", action % state.size)),
                        "raw_policy_prior": item.get("raw_policy_prior"),
                        "search_prior": item.get("search_prior"),
                        "visits": item.get("visits"),
                        "q": item.get("q"),
                        "pi": item.get("pi"),
                        "selected": bool(item.get("selected", False)),
                        "search_config": config,
                    },
                    identity.canonical_key,
                )
            )
            if item.get("q") is not None:
                artifact.add_fact(
                    make_fact(
                        subject_id=candidate.entity_id,
                        predicate=Predicate.HAS_ACTION_VALUE,
                        value=float(item["q"]),
                        provenance_id=mcts_source.provenance_id,
                        epistemic_class=EpistemicClass.LEARNED,
                    )
                )
    return artifact
