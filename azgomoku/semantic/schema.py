"""Serializable entity, fact, provenance, and artifact contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Iterable

from .epistemic import EpistemicClass, epistemic_value
from .predicates import Predicate, predicate_value


class EntityType(str, Enum):
    BoardState = "BoardState"
    Cell = "Cell"
    Move = "Move"
    LineWindow = "LineWindow"
    WinningThreat = "WinningThreat"
    DefenseSet = "DefenseSet"
    ForcedResponse = "ForcedResponse"
    Proof = "Proof"
    ProofNode = "ProofNode"
    StructuralEdge = "StructuralEdge"
    AttentionObservation = "AttentionObservation"
    MCTSCandidate = "MCTSCandidate"


def entity_type_value(value: EntityType | str) -> str:
    return value.value if isinstance(value, EntityType) else str(value)


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: EntityType | str
    state_id: str | None
    attributes: dict[str, Any] = field(default_factory=dict)
    canonical_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.entity_id, str) or not self.entity_id:
            raise ValueError("entity_id must be a non-empty string")
        if not isinstance(self.attributes, dict):
            raise ValueError("attributes must be a dict")

    def dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_type": entity_type_value(self.entity_type),
            "state_id": self.state_id,
            "canonical_key": self.canonical_key,
            "attributes": self.attributes,
        }


@dataclass(frozen=True)
class RelationFact:
    fact_id: str
    subject_id: str
    predicate: Predicate | str
    object_id: str | None
    value: Any | None
    provenance_id: str
    epistemic_class: EpistemicClass | str

    def __post_init__(self) -> None:
        if not isinstance(self.fact_id, str) or not self.fact_id:
            raise ValueError("fact_id must be a non-empty string")
        if not isinstance(self.subject_id, str) or not self.subject_id:
            raise ValueError("subject_id must be a non-empty string")
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be a non-empty string")
        if (self.object_id is None) == (self.value is None):
            raise ValueError("exactly one of object_id or value must be set")

    def dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject_id": self.subject_id,
            "predicate": predicate_value(self.predicate),
            "object_id": self.object_id,
            "value": self.value,
            "provenance_id": self.provenance_id,
            "epistemic_class": epistemic_value(self.epistemic_class),
        }


@dataclass(frozen=True)
class Provenance:
    provenance_id: str
    state_id: str | None
    source_kind: str
    source_file: str | None = None
    source_function: str | None = None
    method: str | None = None
    status: str | None = None
    generator_version: str | None = None
    artifact_ref: str | None = None
    proof_or_certificate_id: str | None = None
    model_checkpoint: str | None = None
    budget: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be a non-empty string")
        if not isinstance(self.source_kind, str) or not self.source_kind:
            raise ValueError("source_kind must be a non-empty string")
        if self.budget is not None and not isinstance(self.budget, dict):
            raise ValueError("budget must be a dict or None")

    def dict(self) -> dict[str, Any]:
        return asdict(self)


class SemanticArtifact:
    """Mutable deterministic assembly container with conflict-safe merging."""

    def __init__(
        self,
        entities: Iterable[Entity] = (),
        facts: Iterable[RelationFact] = (),
        provenance: Iterable[Provenance] = (),
    ) -> None:
        self.entities: dict[str, Entity] = {}
        self.facts: dict[str, RelationFact] = {}
        self.provenance: dict[str, Provenance] = {}
        for item in entities:
            self.add_entity(item)
        for item in provenance:
            self.add_provenance(item)
        for item in facts:
            self.add_fact(item)

    def add_entity(self, entity: Entity) -> Entity:
        existing = self.entities.get(entity.entity_id)
        if existing is None:
            self.entities[entity.entity_id] = entity
            return entity
        if (
            entity_type_value(existing.entity_type) != entity_type_value(entity.entity_type)
            or existing.state_id != entity.state_id
            or existing.canonical_key != entity.canonical_key
        ):
            raise ValueError(f"conflicting entity identity: {entity.entity_id}")
        merged = dict(existing.attributes)
        for key, value in entity.attributes.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"conflicting entity attribute: {entity.entity_id}.{key}")
            merged[key] = value
        if merged != existing.attributes:
            existing = replace(existing, attributes=merged)
            self.entities[entity.entity_id] = existing
        return existing

    def add_fact(self, fact: RelationFact) -> RelationFact:
        existing = self.facts.get(fact.fact_id)
        if existing is not None and existing != fact:
            raise ValueError(f"conflicting fact identity: {fact.fact_id}")
        self.facts[fact.fact_id] = fact
        return fact

    def add_provenance(self, item: Provenance) -> Provenance:
        existing = self.provenance.get(item.provenance_id)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting provenance identity: {item.provenance_id}")
        self.provenance[item.provenance_id] = item
        return item

    def merge(self, other: "SemanticArtifact") -> "SemanticArtifact":
        for item in other.entities.values():
            self.add_entity(item)
        for item in other.provenance.values():
            self.add_provenance(item)
        for item in other.facts.values():
            self.add_fact(item)
        return self

