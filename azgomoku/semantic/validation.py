"""Strict referential and epistemic validation for Semantic KG v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .epistemic import EpistemicClass, epistemic_value
from .predicates import (
    TACTICAL_TRUTH_PREDICATES,
    UNAVAILABLE_PREDICATES,
    VALUE_PREDICATES,
    Predicate,
    predicate_value,
)
from .schema import Entity, EntityType, Provenance, RelationFact, SemanticArtifact, entity_type_value


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    errors: tuple[str, ...]
    entity_count: int
    fact_count: int
    provenance_count: int


class SemanticValidationError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("semantic artifact validation failed: " + "; ".join(self.errors))


def _duplicates(values: list[str]) -> set[str]:
    seen: set[str] = set()
    return {value for value in values if value in seen or seen.add(value)}


def validate_collections(
    entities: Iterable[Entity],
    facts: Iterable[RelationFact],
    provenance: Iterable[Provenance],
    *,
    raise_on_error: bool = False,
) -> ValidationReport:
    entity_items = list(entities)
    fact_items = list(facts)
    provenance_items = list(provenance)
    errors: list[str] = []

    duplicate_entities = _duplicates([item.entity_id for item in entity_items])
    duplicate_facts = _duplicates([item.fact_id for item in fact_items])
    duplicate_provenance = _duplicates([item.provenance_id for item in provenance_items])
    errors.extend(f"duplicate entity_id: {value}" for value in sorted(duplicate_entities))
    errors.extend(f"duplicate fact_id: {value}" for value in sorted(duplicate_facts))
    errors.extend(f"duplicate provenance_id: {value}" for value in sorted(duplicate_provenance))

    entity_map = {item.entity_id: item for item in entity_items}
    provenance_map = {item.provenance_id: item for item in provenance_items}
    allowed_entity_types = {item.value for item in EntityType}
    allowed_predicates = {item.value for item in Predicate}
    allowed_epistemic = {item.value for item in EpistemicClass}

    for entity in entity_items:
        kind = entity_type_value(entity.entity_type)
        if kind not in allowed_entity_types:
            errors.append(f"invalid entity_type for {entity.entity_id}: {kind}")

    for fact in fact_items:
        predicate = predicate_value(fact.predicate)
        epistemic = epistemic_value(fact.epistemic_class)
        if fact.subject_id not in entity_map:
            errors.append(f"dangling subject for {fact.fact_id}: {fact.subject_id}")
        if fact.object_id is not None and fact.object_id not in entity_map:
            errors.append(f"dangling object for {fact.fact_id}: {fact.object_id}")
        source = provenance_map.get(fact.provenance_id)
        if source is None:
            errors.append(f"missing provenance for {fact.fact_id}: {fact.provenance_id}")
        if predicate in UNAVAILABLE_PREDICATES or predicate not in allowed_predicates:
            errors.append(f"invalid predicate for {fact.fact_id}: {predicate}")
        if epistemic not in allowed_epistemic:
            errors.append(f"invalid epistemic class for {fact.fact_id}: {epistemic}")
            continue
        if predicate in VALUE_PREDICATES and fact.value is None:
            errors.append(f"value predicate lacks value for {fact.fact_id}: {predicate}")
        if predicate not in VALUE_PREDICATES and fact.object_id is None:
            errors.append(f"entity predicate lacks object_id for {fact.fact_id}: {predicate}")
        if epistemic == EpistemicClass.HEURISTIC.value:
            errors.append(f"HEURISTIC emission is reserved in Phase 1-3: {fact.fact_id}")
        if source is None:
            continue
        if epistemic == EpistemicClass.EXACT.value:
            if source.method != "full_minimax" or source.status != "exact_complete":
                errors.append(f"EXACT fact lacks exact-complete full-minimax provenance: {fact.fact_id}")
        if predicate == Predicate.OPTIMAL_IN.value and epistemic != EpistemicClass.EXACT.value:
            errors.append(f"OPTIMAL_IN must be EXACT in v1: {fact.fact_id}")
        if epistemic == EpistemicClass.CERTIFIED.value:
            if (
                source.method not in {"tactical_replay", "vcf"}
                or source.status != "replay_passed"
                or not source.proof_or_certificate_id
            ):
                errors.append(f"CERTIFIED fact lacks replay evidence: {fact.fact_id}")
        if epistemic == EpistemicClass.DERIVED.value:
            if not source.source_function and not source.artifact_ref:
                errors.append(f"DERIVED fact lacks deterministic lineage: {fact.fact_id}")
        if epistemic == EpistemicClass.LEARNED.value:
            if predicate in TACTICAL_TRUTH_PREDICATES:
                errors.append(f"LEARNED fact uses tactical truth predicate: {fact.fact_id}:{predicate}")
            if source.source_kind not in {"model_evidence", "mcts"} or not source.model_checkpoint:
                errors.append(f"LEARNED fact lacks model/search provenance: {fact.fact_id}")
            if source.source_kind == "mcts" and not source.budget:
                errors.append(f"MCTS fact lacks search config: {fact.fact_id}")

    report = ValidationReport(
        valid=not errors,
        errors=tuple(errors),
        entity_count=len(entity_items),
        fact_count=len(fact_items),
        provenance_count=len(provenance_items),
    )
    if raise_on_error and errors:
        raise SemanticValidationError(errors)
    return report


def validate_artifact(
    artifact: SemanticArtifact, *, raise_on_error: bool = False
) -> ValidationReport:
    return validate_collections(
        artifact.entities.values(),
        artifact.facts.values(),
        artifact.provenance.values(),
        raise_on_error=raise_on_error,
    )
