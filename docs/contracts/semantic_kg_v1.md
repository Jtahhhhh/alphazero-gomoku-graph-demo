# Semantic KG v1 Contract

Version: `semantic_kg_v1`  
Scope: source-grounded Phase 1-3 only.

## Boundary

Semantic KG v1 is a deterministic derivative of existing Gomoku state, graph, tactics, proof, solver, model-evidence, and MCTS artifacts. It does not change any upstream result.

Generic `Pattern` and `OpenEnd` are out of scope. `EXTENDS`, `HAS_OPEN_END`, `HAS_PATTERN`, and any other undeclared predicate must not be emitted. Phase 1-3 must emit no `HEURISTIC` fact.

## Entities

The closed v1 entity vocabulary is:

`BoardState`, `Cell`, `Move`, `LineWindow`, `WinningThreat`, `DefenseSet`, `ForcedResponse`, `Proof`, `ProofNode`, `StructuralEdge`, `AttentionObservation`, and `MCTSCandidate`.

Every entity contains:

```text
entity_id
entity_type
state_id
canonical_key
attributes
```

`entity_id` is raw/state-scoped identity. `canonical_key` is D4 equivalence identity. Relations must be represented as `RelationFact`, not hidden in entity attributes.

## Facts

Every fact contains exactly one of `object_id` or `value` and always contains `subject_id`, `predicate`, `provenance_id`, and `epistemic_class`. Subject and entity-object IDs must resolve in the same artifact.

The closed predicate vocabulary is:

```text
CONTAINS PLAYED_AT
CREATES BLOCKS USES_CELL HAS_COMPLETION HAS_DIRECTION FORCES
SUPPORTS HAS_WINDOW REQUIRES
OPTIMAL_IN HAS_ACTION_VALUE
CONNECTS HAS_WEIGHT OVERLAPS
```

`HAS_DIRECTION`, `HAS_ACTION_VALUE`, and `HAS_WEIGHT` take literal values. All other v1 predicates take entity objects.

## Epistemic classes

- `EXACT`: only full-minimax facts with `status=exact_complete`. In v1, `OPTIMAL_IN` is always EXACT.
- `CERTIFIED`: only a tactical or VCF claim that passed the existing replay path and names its proof/certificate.
- `DERIVED`: deterministic structure, membership, coordinate transform, or serialization with source lineage.
- `HEURISTIC`: reserved; forbidden in Phase 1-3 exports.
- `LEARNED`: model or MCTS evidence. It may not use tactical-truth predicates such as `CREATES`, `BLOCKS`, `FORCES`, `SUPPORTS`, `REQUIRES`, or `OPTIMAL_IN`.

Proof existence/support is `CERTIFIED`; geometry copied from a replayed proof (`USES_CELL`, `HAS_WINDOW`, `REQUIRES`) remains `DERIVED`.

## Provenance

Every fact resolves to one `Provenance` record. EXACT, CERTIFIED, and LEARNED facts must be traceable respectively to full minimax status, replay evidence, or model/search configuration. Record-level provenance alone is not sufficient unless the fact points to it through `provenance_id`.

## Hard validation gates

An artifact is invalid on duplicate IDs, dangling references, missing provenance, unknown vocabulary, invalid object/value mode, epistemic/provenance mismatch, any HEURISTIC emission, or learned evidence asserting tactical truth.
