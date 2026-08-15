"""Separate contracts for the learned-evidence v1.1 overlay.

The overlay may reference entities in the immutable Semantic KG v1, but it never
copies solver facts into its own files and never extends the tactical ontology.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping

from .epistemic import EpistemicClass, epistemic_value
from .identity import stable_digest
from .predicates import TACTICAL_TRUTH_PREDICATES, UNAVAILABLE_PREDICATES
from .schema import Entity, EntityType, entity_type_value


EVIDENCE_GENERATOR_VERSION = "semantic_evidence_v1.1"


class EvidencePredicate(str, Enum):
    OBSERVES = "OBSERVES"
    HAS_POLICY_PROB = "HAS_POLICY_PROB"
    HAS_STATE_VALUE = "HAS_STATE_VALUE"
    HAS_ATTENTION_WEIGHT = "HAS_ATTENTION_WEIGHT"
    REFERS_TO_MOVE = "REFERS_TO_MOVE"
    HAS_MCTS_PRIOR = "HAS_MCTS_PRIOR"
    HAS_VISITS = "HAS_VISITS"
    HAS_Q = "HAS_Q"
    HAS_SEARCH_PROB = "HAS_SEARCH_PROB"
    IS_SELECTED = "IS_SELECTED"


ENTITY_PREDICATES = frozenset(
    {EvidencePredicate.OBSERVES.value, EvidencePredicate.REFERS_TO_MOVE.value}
)
VALUE_PREDICATES = frozenset(item.value for item in EvidencePredicate) - ENTITY_PREDICATES
NETWORK_PREDICATES = frozenset(
    {EvidencePredicate.HAS_POLICY_PROB.value, EvidencePredicate.HAS_STATE_VALUE.value}
)
ATTENTION_PREDICATES = frozenset(
    {EvidencePredicate.OBSERVES.value, EvidencePredicate.HAS_ATTENTION_WEIGHT.value}
)
MCTS_PREDICATES = frozenset(
    {
        EvidencePredicate.REFERS_TO_MOVE.value,
        EvidencePredicate.HAS_MCTS_PRIOR.value,
        EvidencePredicate.HAS_VISITS.value,
        EvidencePredicate.HAS_Q.value,
        EvidencePredicate.HAS_SEARCH_PROB.value,
        EvidencePredicate.IS_SELECTED.value,
    }
)
FORBIDDEN_EVIDENCE_PREDICATES = frozenset(TACTICAL_TRUTH_PREDICATES) | frozenset(
    UNAVAILABLE_PREDICATES
) | {"HAS_ACTION_VALUE", "HAS_WEIGHT"}


def evidence_predicate_value(value: EvidencePredicate | str) -> str:
    return value.value if isinstance(value, EvidencePredicate) else str(value)


@dataclass(frozen=True)
class EvidenceFact:
    fact_id: str
    subject_id: str
    predicate: EvidencePredicate | str
    object_id: str | None
    value: Any | None
    provenance_id: str
    epistemic_class: EpistemicClass | str

    def __post_init__(self) -> None:
        if not self.fact_id or not self.subject_id or not self.provenance_id:
            raise ValueError("evidence fact IDs must be non-empty")
        if (self.object_id is None) == (self.value is None):
            raise ValueError("exactly one of object_id or value must be set")

    def dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject_id": self.subject_id,
            "predicate": evidence_predicate_value(self.predicate),
            "object_id": self.object_id,
            "value": self.value,
            "provenance_id": self.provenance_id,
            "epistemic_class": epistemic_value(self.epistemic_class),
        }


@dataclass(frozen=True)
class EvidenceProvenance:
    provenance_id: str
    state_id: str
    source_kind: str
    source_file: str
    source_function: str
    method: str
    status: str
    evidence_generator_version: str
    base_kg_manifest_sha256: str
    model_type: str
    network_mode: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_iteration: int
    training_seed: int
    board_size: int
    win_length: int
    layer: str | None = None
    head: str | None = None
    aggregation_method: str | None = None
    edge_id: str | None = None
    playouts: int | None = None
    search_seed: int | None = None
    temperature: float | None = None
    selection_mode: str | None = None
    root_convention_version: int | None = None
    c_puct: float | None = None

    def __post_init__(self) -> None:
        if not self.provenance_id or not self.state_id:
            raise ValueError("evidence provenance IDs must be non-empty")

    def dict(self) -> dict[str, Any]:
        return asdict(self)


class EvidenceOverlay:
    def __init__(
        self,
        entities: Iterable[Entity] = (),
        facts: Iterable[EvidenceFact] = (),
        provenance: Iterable[EvidenceProvenance] = (),
    ) -> None:
        self.entities: dict[str, Entity] = {}
        self.facts: dict[str, EvidenceFact] = {}
        self.provenance: dict[str, EvidenceProvenance] = {}
        for item in entities:
            self.add_entity(item)
        for item in provenance:
            self.add_provenance(item)
        for item in facts:
            self.add_fact(item)

    def add_entity(self, item: Entity) -> Entity:
        existing = self.entities.get(item.entity_id)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting evidence entity: {item.entity_id}")
        self.entities[item.entity_id] = item
        return item

    def add_fact(self, item: EvidenceFact) -> EvidenceFact:
        existing = self.facts.get(item.fact_id)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting evidence fact: {item.fact_id}")
        self.facts[item.fact_id] = item
        return item

    def add_provenance(self, item: EvidenceProvenance) -> EvidenceProvenance:
        existing = self.provenance.get(item.provenance_id)
        if existing is not None and existing != item:
            raise ValueError(f"conflicting evidence provenance: {item.provenance_id}")
        self.provenance[item.provenance_id] = item
        return item


@dataclass(frozen=True)
class EvidenceValidationReport:
    valid: bool
    errors: tuple[str, ...]
    entity_count: int
    fact_count: int
    provenance_count: int
    external_reference_count: int


class EvidenceValidationError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("evidence overlay validation failed: " + "; ".join(self.errors))


def make_evidence_provenance(**fields: Any) -> EvidenceProvenance:
    payload = dict(fields)
    payload.pop("provenance_id", None)
    return EvidenceProvenance(
        provenance_id=f"eprov:{stable_digest(payload)}",
        **payload,
    )


def make_evidence_fact(
    *,
    subject_id: str,
    predicate: EvidencePredicate | str,
    provenance_id: str,
    epistemic_class: EpistemicClass | str,
    object_id: str | None = None,
    value: Any | None = None,
) -> EvidenceFact:
    payload = {
        "subject_id": subject_id,
        "predicate": evidence_predicate_value(predicate),
        "object_id": object_id,
        "value": value,
        "provenance_id": provenance_id,
        "epistemic_class": epistemic_value(epistemic_class),
    }
    return EvidenceFact(fact_id=f"efact:{stable_digest(payload)}", **payload)


def _entity_type(item: Entity | Mapping[str, Any]) -> str:
    if isinstance(item, Entity):
        return entity_type_value(item.entity_type)
    return str(item["entity_type"])


def validate_evidence_overlay(
    overlay: EvidenceOverlay,
    base_entities: Mapping[str, Entity | Mapping[str, Any]],
    *,
    raise_on_error: bool = False,
) -> EvidenceValidationReport:
    errors: list[str] = []
    allowed_predicates = {item.value for item in EvidencePredicate}
    entity_types = {key: _entity_type(value) for key, value in base_entities.items()}
    entity_types.update(
        {key: entity_type_value(value.entity_type) for key, value in overlay.entities.items()}
    )
    provenance = overlay.provenance
    external_references = 0
    facts_by_subject: dict[str, list[EvidenceFact]] = defaultdict(list)
    probability_groups: dict[str, list[float]] = defaultdict(list)
    search_groups: dict[str, list[float]] = defaultdict(list)

    for entity in overlay.entities.values():
        kind = entity_type_value(entity.entity_type)
        if kind not in {EntityType.AttentionObservation.value, EntityType.MCTSCandidate.value}:
            errors.append(f"invalid evidence entity type: {entity.entity_id}:{kind}")

    for item in provenance.values():
        required = (
            item.source_kind,
            item.source_file,
            item.source_function,
            item.method,
            item.status,
            item.evidence_generator_version,
            item.base_kg_manifest_sha256,
            item.model_type,
            item.network_mode,
            item.checkpoint_path,
            item.checkpoint_sha256,
        )
        if any(value in (None, "") for value in required):
            errors.append(f"incomplete evidence provenance: {item.provenance_id}")
        if item.evidence_generator_version != EVIDENCE_GENERATOR_VERSION:
            errors.append(f"wrong evidence generator version: {item.provenance_id}")
        if len(item.checkpoint_sha256) != 64:
            errors.append(f"invalid checkpoint sha256: {item.provenance_id}")
        if item.network_mode != "eval":
            errors.append(f"network evidence not generated in eval mode: {item.provenance_id}")
        if item.source_kind == "attention" and (
            not item.layer or not item.head or not item.aggregation_method or not item.edge_id
        ):
            errors.append(f"attention provenance lacks edge/layer/head metadata: {item.provenance_id}")
        if item.source_kind == "mcts" and (
            item.playouts is None
            or item.search_seed is None
            or item.temperature is None
            or not item.selection_mode
            or item.root_convention_version is None
            or item.c_puct is None
        ):
            errors.append(f"MCTS provenance lacks search configuration: {item.provenance_id}")

    for fact in overlay.facts.values():
        predicate = evidence_predicate_value(fact.predicate)
        epistemic = epistemic_value(fact.epistemic_class)
        facts_by_subject[fact.subject_id].append(fact)
        if predicate in FORBIDDEN_EVIDENCE_PREDICATES or predicate not in allowed_predicates:
            errors.append(f"invalid evidence predicate: {fact.fact_id}:{predicate}")
        if fact.subject_id not in entity_types:
            errors.append(f"dangling evidence subject: {fact.fact_id}:{fact.subject_id}")
        elif fact.subject_id in base_entities:
            external_references += 1
        if fact.object_id is not None:
            if fact.object_id not in entity_types:
                errors.append(f"dangling evidence object: {fact.fact_id}:{fact.object_id}")
            elif fact.object_id in base_entities:
                external_references += 1
        source = provenance.get(fact.provenance_id)
        if source is None:
            errors.append(f"missing evidence provenance: {fact.fact_id}")
            continue
        if predicate in ENTITY_PREDICATES and fact.object_id is None:
            errors.append(f"entity evidence predicate lacks object: {fact.fact_id}")
        if predicate in VALUE_PREDICATES and fact.value is None:
            errors.append(f"value evidence predicate lacks value: {fact.fact_id}")
        expected_epistemic = (
            EpistemicClass.DERIVED.value
            if predicate in ENTITY_PREDICATES
            else EpistemicClass.LEARNED.value
        )
        if epistemic != expected_epistemic:
            errors.append(f"wrong evidence epistemic class: {fact.fact_id}:{epistemic}")
        expected_source = (
            "network" if predicate in NETWORK_PREDICATES
            else "attention" if predicate in ATTENTION_PREDICATES
            else "mcts"
        )
        if source.source_kind != expected_source:
            errors.append(f"wrong evidence source kind: {fact.fact_id}:{source.source_kind}")

        subject_type = entity_types.get(fact.subject_id)
        object_type = entity_types.get(fact.object_id) if fact.object_id else None
        if predicate == EvidencePredicate.OBSERVES.value and (
            subject_type != EntityType.AttentionObservation.value
            or object_type != EntityType.StructuralEdge.value
        ):
            errors.append(f"attention-edge join type mismatch: {fact.fact_id}")
        if predicate == EvidencePredicate.HAS_ATTENTION_WEIGHT.value and (
            subject_type != EntityType.AttentionObservation.value
            or not isinstance(fact.value, (int, float))
            or not math.isfinite(float(fact.value))
            or not 0.0 <= float(fact.value) <= 1.0
        ):
            errors.append(f"invalid attention weight: {fact.fact_id}")
        if predicate == EvidencePredicate.HAS_POLICY_PROB.value:
            if subject_type != EntityType.Move.value or not 0.0 <= float(fact.value) <= 1.0:
                errors.append(f"invalid policy probability: {fact.fact_id}")
            probability_groups[fact.provenance_id].append(float(fact.value))
        if predicate == EvidencePredicate.HAS_STATE_VALUE.value and (
            subject_type != EntityType.BoardState.value
            or not isinstance(fact.value, (int, float))
            or not math.isfinite(float(fact.value))
        ):
            errors.append(f"invalid network state value: {fact.fact_id}")
        if predicate == EvidencePredicate.REFERS_TO_MOVE.value and (
            subject_type != EntityType.MCTSCandidate.value or object_type != EntityType.Move.value
        ):
            errors.append(f"MCTS-move join type mismatch: {fact.fact_id}")
        if predicate in MCTS_PREDICATES - {EvidencePredicate.REFERS_TO_MOVE.value}:
            if subject_type != EntityType.MCTSCandidate.value:
                errors.append(f"MCTS value subject mismatch: {fact.fact_id}")
            if predicate in {
                EvidencePredicate.HAS_MCTS_PRIOR.value,
                EvidencePredicate.HAS_SEARCH_PROB.value,
            } and not 0.0 <= float(fact.value) <= 1.0:
                errors.append(f"invalid MCTS probability: {fact.fact_id}")
            if predicate == EvidencePredicate.HAS_SEARCH_PROB.value:
                search_groups[fact.provenance_id].append(float(fact.value))
            if predicate == EvidencePredicate.HAS_VISITS.value and (
                not isinstance(fact.value, int) or fact.value < 0
            ):
                errors.append(f"invalid MCTS visits: {fact.fact_id}")
            if predicate == EvidencePredicate.IS_SELECTED.value and not isinstance(fact.value, bool):
                errors.append(f"invalid MCTS selected flag: {fact.fact_id}")

    for provenance_id, values in probability_groups.items():
        if abs(sum(values) - 1.0) > 1e-5:
            errors.append(f"policy probabilities do not sum to one: {provenance_id}:{sum(values)}")
    for provenance_id, values in search_groups.items():
        if abs(sum(values) - 1.0) > 1e-5:
            errors.append(f"search probabilities do not sum to one: {provenance_id}:{sum(values)}")
    for entity_id in overlay.entities:
        predicates = {evidence_predicate_value(item.predicate) for item in facts_by_subject[entity_id]}
        kind = entity_types[entity_id]
        if kind == EntityType.AttentionObservation.value and predicates != {
            EvidencePredicate.OBSERVES.value,
            EvidencePredicate.HAS_ATTENTION_WEIGHT.value,
        }:
            errors.append(f"incomplete attention observation: {entity_id}:{sorted(predicates)}")
        if kind == EntityType.MCTSCandidate.value and not {
            EvidencePredicate.REFERS_TO_MOVE.value,
            EvidencePredicate.HAS_MCTS_PRIOR.value,
            EvidencePredicate.HAS_VISITS.value,
            EvidencePredicate.HAS_Q.value,
            EvidencePredicate.HAS_SEARCH_PROB.value,
            EvidencePredicate.IS_SELECTED.value,
        } <= predicates:
            errors.append(f"incomplete MCTS candidate: {entity_id}:{sorted(predicates)}")

    report = EvidenceValidationReport(
        valid=not errors,
        errors=tuple(errors),
        entity_count=len(overlay.entities),
        fact_count=len(overlay.facts),
        provenance_count=len(overlay.provenance),
        external_reference_count=external_references,
    )
    if errors and raise_on_error:
        raise EvidenceValidationError(errors)
    return report
