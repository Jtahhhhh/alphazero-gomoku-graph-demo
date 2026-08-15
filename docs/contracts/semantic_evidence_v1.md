# Semantic Evidence Overlay Contract v1.1

## Scope

`semantic_evidence_v1` is a learned/search observation overlay over the immutable
solver-grounded `semantic_kg` v1. It may reference base entity IDs, but it must
not copy, replace, append to, or reinterpret base solver facts.

The overlay contains no `Pattern`, `OpenEnd`, `EXTENDS`, `HAS_OPEN_END`, or
handcrafted tactical ontology.

## Predicates

Network outputs:

- `Move HAS_POLICY_PROB number`
- `BoardState HAS_STATE_VALUE number`

R-GAT attention:

- `AttentionObservation OBSERVES StructuralEdge`
- `AttentionObservation HAS_ATTENTION_WEIGHT number`

MCTS root evidence:

- `MCTSCandidate REFERS_TO_MOVE Move`
- `MCTSCandidate HAS_MCTS_PRIOR number`
- `MCTSCandidate HAS_VISITS integer`
- `MCTSCandidate HAS_Q number`
- `MCTSCandidate HAS_SEARCH_PROB number`
- `MCTSCandidate IS_SELECTED boolean`

`HAS_ACTION_VALUE` remains reserved for exact solver action values. Network
value, policy probability, MCTS prior and MCTS Q are never serialized through
that predicate. `HAS_WEIGHT` from Semantic KG v1 is not reused; v1.1 uses the
attention-specific `HAS_ATTENTION_WEIGHT`.

## Epistemic boundary

Numeric model/search outputs are `LEARNED`. Exact joins from an observation to a
base entity (`OBSERVES`, `REFERS_TO_MOVE`) are `DERIVED`. The overlay rejects
`EXACT`, `CERTIFIED`, and `HEURISTIC` facts.

Learned evidence may never emit tactical truth predicates, including
`CREATES`, `BLOCKS`, `FORCES`, `SUPPORTS`, `REQUIRES`, or `OPTIMAL_IN`.

## Provenance

Every fact resolves to provenance containing:

- state ID, model type and `eval` network mode;
- checkpoint path, SHA-256 and training iteration;
- training seed, board size and win length;
- evidence generator version and exact source function;
- immutable base KG manifest SHA-256.

Attention provenance additionally contains edge ID, layer, head scope and the
locked aggregation method. The v1.1 observation is one entity per edge and
checkpoint; the stored scalar is the mean across all attention heads, while the
per-head vector remains an entity attribute.

MCTS provenance additionally contains playouts, search seed, temperature,
selection mode, `c_puct`, and root-value convention version. Phase 4 uses the
existing 50-playout schedule at iterations 0, 20, 40, and 60 only, with no root
Dirichlet noise.

## Exact joins

- policy facts use the base state-scoped `Move` ID;
- value facts use the base `BoardState` ID;
- attention observations resolve to the base state-scoped `StructuralEdge` ID;
- MCTS candidates resolve to the base `Move` ID;
- all referenced base IDs must exist in the frozen base files.

Coordinate fuzzy matching is forbidden. An unresolved ID fails the export.

## Node attention convention for Phase 5

Node semantic attention is declared before measurement as **incoming attention
mass**: sum the final-layer mean-head coefficient of every directed edge whose
target is that cell, then normalize across all board cells for mass metrics.
This matches the R-GAT coefficient semantics (incoming messages grouped by
destination and relation). No alternative aggregation is selected post hoc.

## Immutability

The four base files are hashed before Phase 4 and checked again after every
release/export gate:

- `semantic_kg/entities.jsonl`
- `semantic_kg/facts.jsonl`
- `semantic_kg/provenance.jsonl`
- `semantic_kg/manifest.json`

Evidence is written only to `semantic_evidence_v1`. A count/hash change in the
base layer is a hard failure.
