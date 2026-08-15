# Semantic Knowledge Graph Readiness Audit

Audit date: 2026-08-14  
Scope: read-only audit of current source and retained artifacts. No source code, schema, benchmark, renderer, or model artifact was changed.  
Required readiness threshold: a research-grade Semantic KG must reach Level 3 (provenance-aware semantic facts).

## A. Executive verdict

**Strict verdict: PARTIALLY READY.**

**Current fully satisfied level: Level 1 — Annotated Graph.** The repository has a row-major cell graph, typed spatial edges, policy/value output, MCTS traces, learned attention, solver labels, and proof overlays. It has substantial Level 2 ingredients, but it does **not** fully satisfy Level 2 because `Pattern` is not a defined entity and the tactical associations are not materialized as typed relations. It does not satisfy Level 3 because there is no fact store or contract of the form `(subject_id, predicate, object_id/value, provenance_id)`, most semantic objects have no stable IDs, and epistemic classes are not represented by a single fact-level vocabulary.

This is not a vague “potential” verdict. The decisive findings are:

1. `Threat` is explicit as `WinningThreat`; move-created threat structure is explicit as `MoveThreats`; defence is explicit as `DefenseSet` (`azgomoku/tactics.py:17-42`).
2. `Proof` is explicit in two forms: a replayable VCF OR/AND tree (`azgomoku/vcf.py:20-43`) and a flat machine-readable proof annotation with `action`, `concepts`, `critical_cells`, `critical_relations`, and `windows` (`azgomoku/tactics.py:199-225`, `azgomoku/vcf.py:259-294`).
3. `Move` is a machine-readable row-major action and is joined across solver, proof, graph, policy, MCTS, and renderer, but it is not a stable event entity. The strongest existing join gate checks proof action ∈ legal actions ∩ optimal actions ∩ graph nodes and checks D4 coordinate round-trips (`investigation/e3b_graph.py:45-83`).
4. `ForcedResponse` is deterministic but not first-class: it is recoverable from `DefenseSet.blocking_moves` and VCF AND-node children (`azgomoku/tactics.py:116-129`, `azgomoku/vcf.py:145-205`, `azgomoku/vcf.py:243-253`).
5. `Pattern` is **AMBIGUOUS**, not a source-supported generic entity. The source has bounded labels/flags (`immediate_win`, `winning_line`, `mandatory_block`, `simple_fork`, five/four/three/open-three/double-four/four-three), but no `Pattern` class, schema, stable vocabulary contract, or pattern identity (`azgomoku/tactics.py:35-62`, `azgomoku/tactics.py:199-225`).
6. The frozen benchmark is strong evidence, not a KG. It contains 94/94 exact-complete states; 83 states have 243 replayed flat proofs (242 tactical, one VCF), as recorded in `diagnostic/h1_benchmark_v1/manifest.json:8-44`. The records remain nested state/solver/proof documents rather than entity/relation facts. Examples are visible at `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl:2` (two tactical proofs) and `:19` (one VCF proof plus certificate).
7. `knowledge.svg` is explicitly a “proof-flat contrast” renderer, not a graph store (`azgomoku/explanation/rendering/knowledge_svg.py:1-1`). It draws solver windows/critical cells and top-k attention on separate boards (`azgomoku/explanation/rendering/knowledge_svg.py:38-74`, `:111-174`); it never materializes `Threat`, `Pattern`, `Proof`, or their typed relations as addressable KG entities.

### Predefined scoring rule

The following scoring rule was fixed before calculating the result:

- Entity and relation coverage: only `EXPLICIT` counts as current KG-ready coverage. `DERIVABLE` is reported as extraction potential but scores zero here because the fact is not materialized. `HEURISTIC`, `MISSING`, and `AMBIGUOUS` score zero.
- Provenance, identity, and reasoning checklists: `1` = satisfied, `0.5` = partial/local/scoped, `0` = absent.
- Five category percentages are equally weighted. A numeric score cannot override the hard Level 3 requirements: typed facts, stable semantic identity, fact-level provenance, and epistemic-class separation must all exist for `READY`.

| Category | Predefined checklist result | Score |
|---|---:|---:|
| Explicit entity concepts | 16 / 22 | 72.7% |
| Explicit typed/equivalent relations | 6 / 14 | 42.9% |
| Provenance | 7.0 / 10 | 70.0% |
| Stable identity and D4 | 6.0 / 10 | 60.0% |
| Reasoning capabilities | 7.0 / 10 | 70.0% |
| **Equal-weight readiness score** |  | **63.1 / 100 — PARTIAL** |

Deterministic extraction potential is higher than the current KG-ready score: 20/22 requested concepts are explicit or deterministically derivable, and 12/14 requested relations are explicit or deterministically derivable. That difference is precisely the missing semantic materialization layer.

## Audit method and source boundary

Labels used throughout:

- **EXPLICIT** — a named class, schema field, or retained machine-readable artifact directly represents the concept.
- **DERIVABLE** — current sources deterministically produce it without a new tactical rule, but it is not stored as a first-class concept/relation.
- **HEURISTIC** — a new handcrafted taxonomy/rule is needed.
- **MISSING** — current source does not provide enough semantics.
- **AMBIGUOUS** — nearby data exists, but its meaning is insufficiently unified to claim the requested concept.

Primary sources reviewed include all files requested in the task: `azgomoku/graph.py`, `solver.py`, `tactics.py`, `vcf.py`, `ground_truth.py`, `h1_schema.py`, all of `azgomoku/explanation/`, `investigation/e3b5_knowledge.py`, the frozen benchmark JSONL/manifest, the implicit proof/certificate contracts, the knowledge renderer, tactical examples, and tactical tests. Supporting identity/evidence sources reviewed include `game.py`, `symmetry.py`, `mcts.py`, `models/rgat.py`, `models/rgcn.py`, `investigation/e3b_common.py`, `e3b_graph.py`, `e3b_pipeline.py`, `arena_knowledge.py`, and `evaluate_h1.py`.

No standalone proof JSON Schema or certificate JSON Schema exists in the file inventory. Their current contracts are distributed across the `ProofNode` dataclass/serializer/parser/replayer and the flat-proof/certificate producers and replay functions (`azgomoku/vcf.py:32-43`, `azgomoku/h1_schema.py:40-48`, `azgomoku/vcf.py:216-256`, `investigation/e3b_common.py:73-126`, `:152-179`).

## Existing object inventory

This inventory names only objects that actually exist. Rows such as “board entry” explicitly avoid inventing a `Stone` entity.

| Existing object | Semantic meaning in current source | Stable ID? | Main attributes | Relations represented in/out | Exact source |
|---|---|---|---|---|---|
| `GomokuState` | Immutable board state | Artifact-level `state_id`; not stored on class | `board`, `to_play`, `last_move`, `win_length`, derived `size` | legal actions, `play(action)` → next state, winner/outcome | `azgomoku/game.py:4-36`; ID at `azgomoku/explanation/explanation_schema.py:16-18` |
| Board array entry | Cell occupancy value `-1/0/+1`; **not a Stone object** | No | row/col through array position; integer state | belongs to board only by array nesting | `azgomoku/game.py:6-18`; benchmark state board at `azgomoku/h1_schema.py:55-59` |
| Cell/action record | Row-major board location used by graph/policy/action APIs | Local integer `action`; no state/board prefix | `action`, `row`, `col` | graph endpoints; selected move; MCTS candidate | `azgomoku/explanation/explanation_schema.py:11-13` |
| Structural edge record | Directed spatial adjacency of one relation | `relation:source:target`, stable only in a known board size | `edge_id`, `source`, `target`, `relation` | connects two cell/action indices | `azgomoku/graph.py:3-17` |
| `SolverResult` | Full minimax result when exact | No result ID | `status`, `value`, `optimal_actions`, `action_values`, nodes/time | associates action values/optimal actions with input state | `azgomoku/solver.py:12-28`, `:75-95` |
| `GroundTruthResult` | Routed exact-complete, VCF exact-partial, or unknown result | No result ID | method/status/completeness, action values, proof, budget, coverage | state → solver truth/proof | `azgomoku/ground_truth.py:13-42`, `:64-100` |
| `WinningThreat` | One empty completion cell plus a concrete winning window | No | `completion`, `relation`, `window` | threat uses window cells; has a completion | `azgomoku/tactics.py:17-23`, producer at `:104-113` |
| `DefenseSet` | Sound immediate-threat blocking summary | No | `completions`, `blocking_moves`, `unstoppable` | attacker threats imply defender blocks | `azgomoku/tactics.py:26-32`, producer at `:116-129` |
| `MoveThreats` | Tactical primitives created by one legal move | No | `move`, `creates_five`, `fours`, `three_extensions`; derived flags | move creates fours/five/three candidates | `azgomoku/tactics.py:35-62`, producer at `:162-196` |
| Flat proof dict | Action-conditioned geometry/VCF annotation | No proof ID; optional `certificate_id` for VCF only | `action`, `concepts`, `critical_cells`, `critical_relations`, `windows`, method/status | supports one action; uses cells/windows | tactical producer `azgomoku/tactics.py:199-225`; VCF reducer `azgomoku/vcf.py:259-294` |
| `ProofNode` | Replayable VCF OR/AND strategy tree node | No node/proof ID | `player_to_move`, `move`, `node_type`, `children`, `terminal` | contains child move branches; AND requires all defenses | `azgomoku/vcf.py:20-43`, replay at `:216-256` |
| `VCFResult` | One-sided certified win or abstention | No run/result ID | exact-partial status, action set completeness, tree, flat proofs, budget | state → proven action → proof | `azgomoku/vcf.py:46-74`, solver at `:297-350` |
| H1 record | Frozen state + solver + proof document | Yes, `state_id` | state, provenance, solver, `valid_proofs` | document-level containment, not typed KG relations | writer `azgomoku/h1_schema.py:51-67`; validator `:74-160` |
| Proof certificate dict | Serialized VCF proof tree linked to a flat proof | `certificate_id` exists for VCF annotations only | `certificate_id`, `action`, `tree` | flat proof references certificate | `investigation/e3b_common.py:152-169`; frozen example `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl:19` |
| Explanation document | State-specific network/search evidence document | `state_id`; no evidence-document ID | model/checkpoint, state, selected move, network, MCTS, graph evidence | contains model/search/edge evidence | `azgomoku/explanation/explanation_schema.py:21-43` |
| Network prediction block | Policy/value prediction for a state | No prediction ID | `value`, `raw_policy_prior`, `raw_policy_priors` | scores action indices | `azgomoku/explanation/model_evidence.py:38-41`; model heads at `models/common.py:3-6` |
| Graph evidence edge | Structural edge or learned-attention observation | Reuses structural `edge_id`; no evidence ID | relation, source/target cell dicts, head weights, mean attention, layer | connects cells; has learned weight when R-GAT | `azgomoku/explanation/model_evidence.py:21-35` |
| MCTS `Node` | Search action-edge statistics from parent perspective | No | `prior`, `n`, `w`, `q`, children | parent state/action edge → child search subtree | `azgomoku/mcts.py:5-9`, search at `:19-46` |
| MCTS trace candidate | Root action evidence | Local action only | raw/search prior, visits, Q, pi, selected | candidate action in root state | `azgomoku/explanation/mcts_trace.py:9-17` |
| `knowledge.svg` | Visual contrast of flat solver proof vs learned attention | File path/state context only | SVG data attributes for cells, proof indices, edges, weights | presentation overlays; no stored semantic triples | `azgomoku/explanation/rendering/knowledge_svg.py:38-205`; batch exporter `investigation/e3b5_knowledge.py:29-95` |

## B. Entity coverage matrix

“First-class identity” is deliberately separate from concept status. An explicit nested artifact is useful but is not automatically a KG entity.

| Requested concept | Status | Current representation and scope | First-class semantic identity? | Exact source |
|---|---|---|---|---|
| `BoardState` | **EXPLICIT** | `GomokuState`; H1 `state` object and `state_id` | Yes at artifact level; raw-orientation hash | `azgomoku/game.py:4-36`; `azgomoku/h1_schema.py:51-59` |
| `Cell` | **EXPLICIT** | row-major graph node/action and `{action,row,col}` record | Local only; not `state_id:rNcM` | `azgomoku/graph.py:5-17`; `azgomoku/explanation/explanation_schema.py:11-13` |
| `Stone` | **DERIVABLE** | occupancy is `board[row,col] ∈ {-1,0,1}` | No Stone object/ID | `azgomoku/game.py:6-18` |
| `Player` | **EXPLICIT** | `to_play/current_player/player_to_move` integer with checked `±1` domain in tactics | Domain value, not entity | `azgomoku/game.py:7-8`; `azgomoku/tactics.py:84-92`; `azgomoku/vcf.py:33-38` |
| `Move` | **EXPLICIT** | integer action; selected move/candidate cell dict; proof node move | Local coordinate, not move-event ID | `azgomoku/game.py:14-19`; `azgomoku/explanation/explanation_schema.py:34-34`; `azgomoku/vcf.py:33-38` |
| `Line` | **DERIVABLE** | deterministic `(relation, tuple[cells])` windows; flat proof stores windows | No Line ID | `azgomoku/tactics.py:65-81`; proof windows at `:215-223` |
| `Pattern` | **AMBIGUOUS** | bounded concept strings and move flags exist; no generic Pattern contract or identity | No | `azgomoku/tactics.py:35-62`, `:215-223`; legacy counts `diagnostic/h1_tactical.summary.json:10-19` |
| `OpenEnd` | **MISSING** | `WinningThreat.completion` is a winning completion and may be an internal gap; it is not defined as a geometric open end | No | threat fields `azgomoku/tactics.py:17-23`; no OpenEnd producer found |
| `Threat` | **EXPLICIT** | `WinningThreat`; `MoveThreats.fours`; immediate-threat tuples | No threat ID; state/player context often only in caller | `azgomoku/tactics.py:17-23`, `:74-81`, `:104-113`, `:132-170` |
| `Defence` | **EXPLICIT** | `DefenseSet` | No | `azgomoku/tactics.py:26-32`, `:116-129` |
| `Block` | **EXPLICIT** | `blocking_moves` and flat proof concept `mandatory_block` | No Block ID; blocked threat is implicit by windows/completions | `azgomoku/tactics.py:116-129`, `:201-218` |
| `Fork / DoubleThreat` | **EXPLICIT** | three distinct scoped semantics: `simple_fork`, `creates_double_four`, VCF `unstoppable_double_threat` | No unified fork/threat identity | `azgomoku/tactics.py:56-62`, `:219-223`; `azgomoku/vcf.py:156-163` |
| `WinningPath` | **DERIVABLE** | paths/strategy branches can be traversed from a VCF `ProofNode`; full minimax stores values but no principal variation | No | VCF tree `azgomoku/vcf.py:32-43`, traversal `:268-282`; minimax result `azgomoku/solver.py:15-28` |
| `ExactSolution` | **EXPLICIT** | `SolverResult(status="exact")`; routed `exact_complete`; `exact_partial` kept distinct | No solution ID | `azgomoku/solver.py:15-28`, `:75-91`; `azgomoku/ground_truth.py:77-97` |
| `VCF` | **EXPLICIT** | named method, result, proof tree, replay, abstention semantics | No VCF run ID | `azgomoku/vcf.py:46-74`, `:216-256`, `:297-350` |
| `Proof` | **EXPLICIT** | VCF tree plus flat proof dict | Only optional VCF `certificate_id`; no general proof ID | `azgomoku/vcf.py:32-43`; `azgomoku/tactics.py:215-223`; `investigation/e3b_common.py:159-165` |
| `ForcedResponse` | **DERIVABLE** | singleton/multiple required blocks and VCF AND children | No standalone object/ID | `azgomoku/tactics.py:116-129`; `azgomoku/vcf.py:145-205`, `:243-253` |
| `OptimalAction` | **EXPLICIT** | `optimal_actions` plus complete/incomplete flag and action map | Action local to state; no classification entity ID | `azgomoku/solver.py:19-20`, `:89-91`; `azgomoku/ground_truth.py:20-33` |
| `PolicyPrediction` | **EXPLICIT** | masked probability vector and serialized raw priors | No prediction ID/generator version on block | `azgomoku/mcts.py:11-17`; `azgomoku/explanation/model_evidence.py:38-41` |
| `AttentionEvidence` | **EXPLICIT** | per-edge, per-head final-layer R-GAT alpha and mean | Reuses edge ID; no observation ID | `models/rgat.py:11-20`, `:29-35`; `azgomoku/explanation/model_evidence.py:27-35` |
| `StructuralRelation` | **EXPLICIT** | four directed adjacency types with stable size-local edge IDs | Edge ID is stable in board-size context | `azgomoku/graph.py:3-17` |
| `MCTSEvidence` | **EXPLICIT** | root candidate priors, visits, Q, pi, selected, convention version | Local action only; no trace/candidate ID | `azgomoku/explanation/mcts_trace.py:5-17` |

### Focus answer: Threat, Pattern, ForcedResponse, Proof, Move

| Concept | How structured is it now? | Bottom line |
|---|---|---|
| `Threat` | Strongest of the tactical concepts: frozen dataclass, completion, relation, concrete window; move classification nests fours and extension moves. | Machine-readable and deterministic, but no ID, no direct state/player provenance, and no general threat taxonomy. |
| `Pattern` | Only bounded labels/boolean properties generated by particular detectors. | Not a structured generic concept; **AMBIGUOUS** until a source-bounded pattern vocabulary is declared. |
| `ForcedResponse` | Mandatory blocks and AND children are explicit coordinates in deterministic structures. | Deterministically extractable, but not a first-class entity or named relation. |
| `Proof` | Replayable VCF tree and flat proof annotations are both machine-readable; frozen benchmark records replay success. | Substantially explicit, but general proof IDs and fact-level provenance are missing; 242/243 frozen proofs are flat tactical proofs without certificates. |
| `Move` | Shared row-major integer and `{action,row,col}` records cross all modules. | Explicit coordinate/action, but not an event entity with stable identity across state transition, proof, and D4. |

## C. Relation coverage matrix

“Deterministic” means the current function/field determines the relation without a learned or handcrafted semantic rule. “Provenance” describes what is available now, not what could be added later.

| Requested relation | Coverage | Subject → object | Exact function/field/artifact | Deterministic? | Current provenance |
|---|---|---|---|---|---|
| `PLAYED_AT` | **DERIVABLE** | Move → Cell | action is itself the row-major cell; `cell(action,size)` supplies row/col | Yes | state context exists; no move/cell IDs or fact record (`azgomoku/explanation/explanation_schema.py:11-13`) |
| `CONTAINS` | **DERIVABLE** | BoardState → Cell/board entry; Proof → child/window | nested board arrays and `ProofNode.children` | Yes | document containment only (`azgomoku/h1_schema.py:55-67`; `azgomoku/vcf.py:32-43`) |
| `CREATES` | **EXPLICIT** (nested equivalent) | Move → WinningThreat/five/three candidate | `MoveThreats.move`, `.fours`, `.creates_five`, `.three_extensions` | Yes | caller state/player not stored on object; no fact ID (`azgomoku/tactics.py:35-62`, `:162-170`) |
| `BLOCKS` | **DERIVABLE** | Move → immediate Threat/window | `DefenseSet.blocking_moves`; `mandatory_block` flat proof action/windows | Yes, for immediate threats only | proof method/status may exist; no blocked-threat ID (`azgomoku/tactics.py:116-129`, `:201-218`) |
| `EXTENDS` | **AMBIGUOUS** | Move → Line | `three_extensions` means a next attacker move would create a four; it does not identify a Line being extended | Candidate computation is deterministic; requested semantics are not | none (`azgomoku/tactics.py:142-159`) |
| `USES_CELL` | **EXPLICIT** (nested equivalent) | Threat/Proof → Cell | `WinningThreat.window`; flat `critical_cells`/`windows` | Yes | flat proof method/status; no threat/proof/cell IDs (`azgomoku/tactics.py:17-23`, `:215-223`) |
| `HAS_OPEN_END` | **MISSING** | Threat/Line → Cell | no OpenEnd definition or producer | No current semantic contract | none |
| `FORCES` | **DERIVABLE** | Threat/attacker move → ForcedResponse | `mandatory_defenses`; AND-node defender branches | Yes within immediate-threat/VCF scope | replayable VCF certificate if present; otherwise nested fields (`azgomoku/tactics.py:116-129`; `azgomoku/vcf.py:145-205`) |
| `SUPPORTS` | **EXPLICIT** (nested equivalent) | Flat Proof → Move | each proof dict has `action`; certificate also has `action` | Yes | `proof_method`, `proof_status`, optional certificate; no general proof ID (`azgomoku/vcf.py:286-294`; `investigation/e3b_common.py:159-165`) |
| `REQUIRES` | **DERIVABLE** | Proof/AND node → ForcedResponse | AND-node children must match `_required_and_moves` during replay | Yes | VCF replay result, but no emitted relation fact (`azgomoku/vcf.py:209-213`, `:243-253`) |
| `OPTIMAL_IN` | **EXPLICIT** (nested equivalent) | Move → BoardState | record `state_id` + `solver.optimal_actions`; completeness flag qualifies semantics | Yes when exact-complete; existential only when exact-partial | solver method/status/budget at record level (`azgomoku/h1_schema.py:51-67`; `azgomoku/ground_truth.py:77-97`) |
| `CONNECTS` | **EXPLICIT** | Structural/attention edge → Cell × Cell | `source`, `target`, `relation`, `edge_id` | Yes | structural generator implicit; model evidence adds layer/attention semantics (`azgomoku/graph.py:16-17`; `azgomoku/explanation/model_evidence.py:21-35`) |
| `HAS_WEIGHT` | **EXPLICIT** | Attention edge → numeric value/head values | `attention`, `head_attention` | Learned observation, deterministically serialized from a model forward | model/checkpoint at document level; no evidence-observation ID (`azgomoku/explanation/model_evidence.py:27-41`) |
| `OVERLAPS` | **DERIVABLE** | Attention/structural edge → Proof element | edge is critical iff relation matches and both endpoints are in one proof window | Yes | metric/evaluator context only; not persisted as relation (`investigation/evaluate_h1.py:44-53`; renderer equivalent `azgomoku/explanation/rendering/knowledge_svg.py:18-30`) |

### Equivalent relations already encoded

These are source-supported associations, not proposed literature concepts:

| Existing equivalent | Source representation | Exact source |
|---|---|---|
| Threat `HAS_COMPLETION` Cell | `WinningThreat.completion` | `azgomoku/tactics.py:17-23` |
| Threat/Line `HAS_DIRECTION` relation | `WinningThreat.relation`, proof `critical_relations` | `azgomoku/tactics.py:17-23`, `:215-223` |
| Proof `HAS_WINDOW` Cell-list | flat proof `windows` | `azgomoku/tactics.py:215-223`; `azgomoku/vcf.py:286-294` |
| ProofNode `HAS_CHILD` ProofNode | `children` | `azgomoku/vcf.py:32-43` |
| Move `HAS_ACTION_VALUE` value | solver `action_values` | `azgomoku/solver.py:19-20`, `:82-91` |
| Result `HAS_STATUS/METHOD/COMPLETENESS` | ground-truth/VCF fields | `azgomoku/ground_truth.py:20-33`; `azgomoku/vcf.py:47-60` |

No audited path emits any of these as a generic typed triple with subject/object IDs.

## D. Provenance matrix

### Can the source emit `(subject, predicate, object, provenance)` now?

**Only partially.** It can usually determine the semantic payload and record-level source context, but it cannot consistently name the subject/object semantic entities or attach a provenance pointer to each individual fact.

| Provenance requirement | Status | Evidence / gap |
|---|---|---|
| `state_id` | **Yes** | raw-state SHA-256 prefix via `state_identifier` (`azgomoku/explanation/explanation_schema.py:16-18`) |
| Generator version, seed, history, ply, dedup | **Yes** | H1 `provenance` writer (`azgomoku/h1_schema.py:60-64`) and richer frozen records, e.g. JSONL `:2` |
| Immutable benchmark version/hash | **Yes** | hash/version/immutable rule (`diagnostic/h1_benchmark_v1/manifest.json:2-4`, `:22-23`) |
| Solver method/status/completeness/perspective | **Yes** | required/validated fields (`azgomoku/h1_schema.py:14-20`, `:103-141`) |
| Solver budget/nodes/elapsed | **Yes** | ground-truth result and validation (`azgomoku/ground_truth.py:29-33`; `azgomoku/h1_schema.py:110-111`) |
| Flat proof method/status | **Yes** | tactical annotations add `tactical_replay`/`exact`; VCF reducer adds `vcf`/`exact` (`investigation/e3b_common.py:145-149`; `azgomoku/vcf.py:286-294`) |
| Proof/certificate source link | **Partial** | only VCF flat proofs receive `certificate_id`; the frozen set has one VCF certificate while 242 tactical proofs have no proof IDs/certificates (`diagnostic/h1_benchmark_v1/manifest.json:39-44`; `investigation/e3b_common.py:152-179`) |
| Learned model/checkpoint attribution | **Partial** | explanation document records model/checkpoint and benchmark provenance can include checkpoint/hash, but an individual attention edge has no evidence ID or generator version (`azgomoku/explanation/explanation_schema.py:24-27`; `azgomoku/explanation/model_evidence.py:27-41`; frozen JSONL `:2`) |
| Fact-level provenance pointer | **No** | no `fact_id`/`provenance_id`; provenance is record/document scoped |
| Unified epistemic class per fact | **No** | solver statuses, proof statuses, and `evidence_kind` exist in separate contracts; no common EXACT/CERTIFIED/DERIVED/HEURISTIC/LEARNED enum |

Predefined provenance score: `1+1+1+1+1+1+0.5+0.5+0+0 = 7.0/10`.

### Required epistemic separation

Current code already contains the evidence needed to preserve the following boundary, but it does not serialize the boundary at fact level:

| Required class | Source-supported payload | Current marker | Audit interpretation |
|---|---|---|---|
| `EXACT` | full minimax value/all optimal root actions | `exact_complete`, `full_minimax`, `optimal_actions_complete=true` | Solver truth |
| `CERTIFIED` | replayed VCF existential win; replayed tactical proof | `exact_partial` + VCF proof replay; or `proof_method=tactical_replay`, `proof_status=exact` | Certified action/proof, with scope stated explicitly |
| `DERIVED` | cells, windows, structural edges, membership and overlap relations | deterministic functions, usually no status | Must not be promoted to solver truth |
| `HEURISTIC` | future/manual pattern taxonomy or rules not present now | no unified marker | No current fact should silently receive this class |
| `LEARNED` | policy/value and R-GAT attention | `evidence_kind=learned_attention`; attention disclaimer | Model evidence, never tactical truth |

The separation is semantically acknowledged in source: attention is explicitly “not causal explanations” (`azgomoku/explanation/explanation_schema.py:39-42`), R-GCN is structural while R-GAT is learned (`azgomoku/explanation/model_evidence.py:34-37`), and solver results are distinct from explanation proofs (`docs/contracts/h1_correctness.md:3-9`). The missing piece is a shared fact-level type system.

## E. Identity and D4 audit

### E1. Stable identity checklist

| Identity check | Score | Finding and exact source |
|---|---:|---|
| Board-state identity | 1 | `state_identifier` hashes raw board/player/last move/win length (`azgomoku/explanation/explanation_schema.py:16-18`) |
| Cell identity | 0.5 | consistent row-major `action`, but no board/state prefix (`azgomoku/explanation/explanation_schema.py:11-13`) |
| Move/action identity | 0.5 | shared integer joins solver/proof/policy/MCTS, but denotes a coordinate, not a move event |
| Proof identity | 0 | flat proofs and `ProofNode`s lack IDs; renderer creates transient `P1`, `P2` from list order (`azgomoku/explanation/rendering/knowledge_svg.py:69-72`, `:190-200`) |
| Certificate identity | 0.5 | present for VCF annotation only (`investigation/e3b_common.py:159-165`) |
| Structural edge identity | 1 | stable `relation:source:target` (`azgomoku/graph.py:16-17`) |
| Attention/structural edge join identity | 1 | both model families reuse structural edge IDs; renderer rejects set mismatch (`azgomoku/explanation/model_evidence.py:21-35`; `azgomoku/explanation/rendering/knowledge_svg.py:51-55`) |
| MCTS candidate identity | 0.5 | local action joins root children/legal/proof actions, but no candidate/trace ID (`azgomoku/explanation/mcts_trace.py:9-17`; `investigation/e3b_graph.py:87-101`) |
| D4 coordinate/proof round-trip | 1 | board/action/relation/flat proof transforms exist and round-trip gate checks all eight symmetries (`azgomoku/symmetry.py:13-69`, `:72-87`; `investigation/e3b_graph.py:45-83`) |
| D4 canonical semantic identity | 0 | no canonical IDs for Threat, Line, Pattern, Block, ForcedResponse, Proof, or Move event |

Predefined identity score: `6.0/10`.

### E2. Same cell across modules

Within a known board size and state, the same integer action is consistently used by:

- board flattening and `play` (`azgomoku/game.py:14-19`);
- cell graph endpoints (`azgomoku/graph.py:5-17`);
- solver action maps (`azgomoku/solver.py:19-20`, `:82-91`);
- flat proof action/cells/windows (`azgomoku/tactics.py:215-223`);
- policy vector index (`azgomoku/mcts.py:11-17`);
- MCTS root child key and trace candidate (`azgomoku/mcts.py:20-25`; `azgomoku/explanation/mcts_trace.py:11-17`);
- renderer `data-action` and edge endpoints (`azgomoku/explanation/rendering/knowledge_svg.py:111-174`).

The coordinate gate makes this an enforced join convention, not merely a coincidence (`investigation/e3b_graph.py:50-76`). However, action `23` alone is not globally unique: it does not encode board size, state, player, ply, or whether it is a candidate, historical move, proof move, or MCTS edge. A Level 3 entity ID must at least scope it to `state_id`, and a move event must also distinguish the transition/ply.

### E3. D4 canonicalization

There are two separate mechanisms:

1. `canonical_key(state)` chooses the minimum among eight transformed raw state keys for deduplication (`azgomoku/symmetry.py:62-69`). The frozen manifest records `dedup_mode="d4"` (`diagnostic/h1_benchmark_v1/manifest.json:8-9`).
2. `transform_flat_proof` transforms action, critical cells, relations, and windows, and the gate round-trips every proof/action through all eight symmetries (`azgomoku/symmetry.py:53-59`; `investigation/e3b_graph.py:66-83`). With 243 proofs and eight transforms, the retained gate scope is 1,944 proof round-trips (`diagnostic/h1_benchmark_v1/manifest.json:39-40`; multiplication defined at `investigation/e3b_graph.py:82-82`).

Neither mechanism provides canonical semantic entity identity. `state_identifier` hashes the raw orientation rather than the D4 canonical key, `transform_flat_proof` transforms content rather than IDs, and no corresponding transform/canonicalizer exists for `WinningThreat`, `MoveThreats`, `DefenseSet`, `ProofNode`, Pattern, or future semantic facts. Thus a rotated `Threat_A` cannot currently be recognized by ID as the same semantic object.

## F. Reasoning-rule inventory

### Predefined reasoning checklist

| Capability | Score | Current source |
|---|---:|---|
| Exact optimal-action inference | 1 | full root action values and maxima (`azgomoku/solver.py:75-91`) |
| Immediate winning-threat inference | 1 | window scan + replay (`azgomoku/tactics.py:74-113`) |
| Mandatory-defense inference | 1 | completion cardinality/block set (`azgomoku/tactics.py:116-129`) |
| Five/four/three move classification | 1 | `classify_threat_move` (`azgomoku/tactics.py:132-170`) |
| Fork/double-threat inference | 1 | double-four property, simple-fork detector, VCF terminal (`azgomoku/tactics.py:56-62`, `:219-223`; `azgomoku/vcf.py:156-163`) |
| Forced VCF OR/AND inference | 1 | `_prove_or`, `_prove_and` (`azgomoku/vcf.py:123-205`) |
| Proof verification/replay | 1 | VCF tree replay and flat proof replay (`azgomoku/vcf.py:216-256`; `investigation/e3b_common.py:73-126`) |
| Generic Pattern/OpenEnd reasoning | 0 | no source-supported generic contract |
| Provenance-aware fact inference | 0 | rules return objects/dicts, not facts with provenance IDs |
| Strategy-explanation chaining beyond VCF | 0 | no rule graph from tactical facts to strategy explanation; exact solver retains no PV/proof |

Reasoning score: `7.0/10`.

### A. Rules already present in code

1. A length-`k` window with exactly `k-1` player stones and one empty cell is an immediate threat; the completion is replay-verified (`azgomoku/tactics.py:74-113`).
2. One distinct completion yields the mandatory ordinary block; two or more distinct completions are unstoppable by one block (`azgomoku/tactics.py:116-129`).
3. A legal move is classified by immediate five, created fours, and next-attacker four-creating extensions; open-three/double-four/four-three flags are deterministic properties of that classification (`azgomoku/tactics.py:132-170`, `:44-62`).
4. A move whose child has at least two distinct immediate winning completion cells yields the current `simple_fork` flat proof (`azgomoku/tactics.py:219-223`).
5. Full minimax determines exact state value, every root action value, and all optimal root actions (`azgomoku/solver.py:75-91`).
6. VCF OR nodes choose attacker five/four candidates; AND nodes enumerate all mandatory defenses, reject counter-wins, and prove only if all defender branches continue (`azgomoku/vcf.py:123-205`).
7. A VCF certificate is accepted only after fail-closed replay, including exact required AND-child sets (`azgomoku/vcf.py:216-256`, `:309-328`).

### B. Rules deterministically expressible from existing artifacts

These require serialization/materialization code, not new Gomoku research:

- `Move PLAYED_AT Cell` from `action` and board size.
- `BoardState CONTAINS Cell` and deterministic board-entry/stone facts.
- `Line USES_CELL Cell` from `windows()`; `Threat USES_CELL Cell` and `HAS_COMPLETION` from `WinningThreat`.
- `Move CREATES Threat` from `MoveThreats.fours` and `creates_five`.
- `Move BLOCKS Threat` from `DefenseSet.blocking_moves` plus the immediate winning windows it answers.
- `Proof SUPPORTS Move`, `Proof USES_CELL Cell`, and `Proof HAS_WINDOW Line` from flat proof fields.
- `Proof REQUIRES ForcedResponse` and `Threat FORCES ForcedResponse` from VCF AND nodes/mandatory defenses.
- `Move OPTIMAL_IN BoardState` from exact-complete solver records; exact-partial actions must be marked certified existential actions, not a complete optimum set.
- `WinningPath` instances from root-to-terminal VCF branches, scoped explicitly to VCF certificates.
- `AttentionEdge OVERLAPS ProofElement` from the current critical-edge predicate (`investigation/evaluate_h1.py:44-53`).
- epistemic separation of solver, deterministic structure, certified proof, and learned evidence using markers already present in separate source contracts.

### C. Rules requiring new research or handcrafted semantics

- a generic `Pattern` ontology beyond the existing bounded labels/flags;
- general `OpenEnd` semantics and identity (a completion cell is not always an endpoint);
- precise `EXTENDS` semantics that identifies the Line/Pattern being extended;
- equivalence/deduplication of overlapping threat windows into one semantic Threat;
- a single semantic definition that unifies or distinguishes `simple_fork`, `double_four`, and VCF `unstoppable_double_threat`;
- canonical semantic identity under D4 for lines, patterns, threats, blocks, responses, and proofs;
- whole-game `WinningPath`/strategy explanations for full minimax states, because the exact solver retains action values but no proof/PV tree;
- higher-level strategy rules from tactical facts to human explanation.

The majority of low-level tactical reasoning is in group A, but the ontology, identity, and higher-level semantic reasoning needed for a full Reasoning KG remain in group C.

## G. Current graph and `knowledge.svg` versus a Semantic KG

### What the renderer actually displays

| Visualization behavior | Structured input currently used | What a true Semantic KG would additionally require |
|---|---|---|
| Board cells/stones | `state.board`, row-major actions | stable Cell/Stone facts scoped to BoardState |
| Orange critical cells | union of flat proof `critical_cells` | Proof/Threat/Line entity IDs plus `USES_CELL` facts |
| Orange proof window lines and labels | proof `windows`, `critical_relations`, `concepts`; transient proof index | persistent Proof and Line/Pattern IDs; `HAS_WINDOW`, `SUPPORTS`, `CONTAINS` facts |
| `P1`, `P2`, … marker | list enumeration in renderer | stable proof identity; list order must not be identity |
| Blue top-k R-GAT edges | evidence edges ranked by `attention`, tied by `edge_id` | AttentionEvidence observation ID, model/checkpoint/generator provenance, `HAS_WEIGHT` fact |
| Dashed tactic reference on attention board | copied `critical_cells` | explicit separation between certified proof facts and learned evidence |
| top-k ∩ proof-edge count | runtime intersection from relation + window membership | materialized `OVERLAPS` facts with derivation provenance |
| status/collapse/alignment captions | solver flags and aggregate metrics | typed fact status and provenance, queryable independently of SVG |

Exact renderer evidence:

- It requires a proof-bearing record and a green D4 graph gate (`azgomoku/explanation/rendering/knowledge_svg.py:38-46`).
- It only checks R-GAT/R-GCN structural edge-ID set equality; structural edges are not rendered as semantic knowledge (`:51-55`, `:145-170`, `:186-186`).
- It derives proof-critical edge IDs at render time from `critical_relations` and `windows` (`:18-30`, `:66-74`).
- It paints critical cells and window lines, not threat/pattern/proof nodes (`:111-143`).
- It paints learned attention separately and states that model evidence is not causal proof (`:145-174`, `:203-203`).
- It assigns proof labels from array position (`:69-72`, `:190-200`).
- Batch rendering produces 83 contrast SVGs and marks 11 no-proof states out of scope; it expects exactly those counts (`investigation/e3b5_knowledge.py:54-91`), matching the frozen manifest’s 83 proof-bearing states (`diagnostic/h1_benchmark_v1/manifest.json:29-44`).

Therefore:

> **visual knowledge ≠ Semantic KG.**

The current artifact is a faithful, useful, proof-flat evidence visualization. It is not a store of semantic entities and typed, provenance-bearing relations. In particular, a displayed orange window does not imply that a `Threat` or `Line` entity exists; a blue edge does not imply tactical truth; and the overlap count is a derived metric rather than an `OVERLAPS` fact.

## H. Readiness level

| Level | Definition | Current status | Evidence |
|---|---|---|---|
| Level 0 — Spatial Graph | cell topology | **Satisfied** | four spatial relations and stable size-local edge IDs (`azgomoku/graph.py:3-17`) |
| Level 1 — Annotated Graph | policy/value/attention/proof overlays | **Satisfied** | explanation schema, model evidence, MCTS trace, proof-flat renderer |
| Level 2 — Semantic Tactical Graph | Pattern, Threat, Block, Move, Proof and typed relations | **Partial, not satisfied** | Threat/Block/Move/Proof data exists, but Pattern is ambiguous and relations are mostly nested/derived |
| Level 3 — Provenance-aware Semantic KG | facts + provenance + epistemic status | **Not satisfied** | strong record provenance exists, but no fact IDs/triples, semantic IDs, or unified EXACT/CERTIFIED/DERIVED/HEURISTIC/LEARNED typing |
| Level 4 — Reasoning KG | rule/inference over facts to strategy explanations | **Not satisfied** | tactical algorithms exist, but no KG rule layer or strategy explanation chain |

**Assigned level: Level 1.** Level 2 is partially implemented in source objects; Level 3 is not present.

### Hard Level 3 gate

| Mandatory Level 3 condition | Pass? |
|---|---|
| Explicit semantic entity records | Partial |
| Typed semantic relation records | No |
| Stable semantic IDs | No |
| Fact-level provenance | No |
| Solver/certified/derived/heuristic/learned separation per fact | No |
| Serialization/query artifact independent of SVG | No |

Because this hard gate fails, the strict verdict remains **PARTIALLY READY**, regardless of the 63.1/100 extraction-readiness score.

## I. Minimum implementation delta to reach Level 3

Do not rewrite the game, solver, tactics, VCF, model, MCTS, benchmark, or SVG renderer. Add one thin semantic adapter/export layer and preserve all current source contracts.

### Minimal module surface

```text
azgomoku/semantic/
├── schema.py             # Entity, RelationFact, Provenance, epistemic enums
├── identity.py           # state-scoped and D4-canonical semantic IDs
├── extract_state.py      # BoardState/Cell/Stone/Move/Line materialization
├── extract_tactics.py    # adapters over current tactics outputs
├── extract_proofs.py     # flat proof + VCF tree/certificate adapters
├── extract_evidence.py   # policy/attention/structural/MCTS, always LEARNED/DERIVED
└── export_kg.py          # deterministic JSON/JSONL export and validation
```

This is seven small adapters/contracts, not a replacement pipeline.

### Minimal source-bounded schema contract

1. `Entity {entity_id, entity_type, state_id, attributes}`.
2. `RelationFact {fact_id, subject_id, predicate, object_id | value, provenance_id, epistemic_class}`.
3. `Provenance {provenance_id, state_id, source_kind, method, status, generator_version, artifact_ref, proof_or_certificate_id, model_checkpoint, budget}`.
4. `EpistemicClass = EXACT | CERTIFIED | DERIVED | HEURISTIC | LEARNED`.
5. Initial entity types only from supported source concepts: `BoardState`, `Cell`, `Move`, `LineWindow`, `WinningThreat`, `DefenseSet`, `Proof`, `ProofNode`, `ForcedResponse`, `StructuralEdge`, `AttentionObservation`, `MCTSCandidate`. Do **not** claim a generic `Pattern` or `OpenEnd` entity until their contracts are defined.
6. Initial predicates only from explicit/deterministic rows in section C. Mark `EXTENDS` and `HAS_OPEN_END` unavailable until semantic definitions exist.

### Identity minimum

- Cell: `cell:{state_id}:r{row}c{col}` (or board-template ID plus state membership; choose one and validate consistently).
- Move candidate: `move:{state_id}:a{action}`; historical move event additionally includes ply/next-state ID.
- Structural edge: retain existing `relation:source:target` as a legacy key, but scope the semantic ID by board size or state.
- Flat proof: deterministic hash over canonical normalized proof content plus `state_id`; do not use transient `P1` order.
- VCF node: proof ID plus a deterministic child path.
- Threat/line/response: deterministic content hash over state, player, relation, ordered window/completion/response fields.
- Add a D4 canonical-key field separately from the raw-orientation entity ID. Verify semantic IDs/transforms, not only coordinate round-trips.

### Extraction minimum

1. Materialize `PLAYED_AT`, `CONTAINS`, `USES_CELL`, `SUPPORTS`, `OPTIMAL_IN`, `CONNECTS`, and `HAS_WEIGHT` directly.
2. Materialize `CREATES`, `BLOCKS`, `FORCES`, `REQUIRES`, and `OVERLAPS` only through the exact current functions identified in section C; record function/version in provenance.
3. Export full-minimax optimal facts as `EXACT` only when `exact_complete` and `optimal_actions_complete=true`.
4. Export replayed VCF support as `CERTIFIED` with exact-partial scope; never claim its action set is complete.
5. Export geometry/structure membership as `DERIVED`; export policy/value/attention/MCTS as `LEARNED` (MCTS also records search configuration). Never merge attention into Threat/Proof facts.
6. For the 11/94 frozen no-proof states, export exact solver facts but no tactical Proof/Threat facts. Preserve the current honest abstention behavior.

### Two small semantic definitions still required

To fully satisfy Level 2/3 as defined by the task, two bounded contracts must be added:

1. **Pattern vocabulary:** restrict v1 to current source-supported labels/properties only. Declare whether `winning_line`, `immediate_win`, `mandatory_block`, `simple_fork`, `double_four`, `four_three`, and VCF `forced_sequence` are Pattern types, proof concepts, or both. Do not import additional Gomoku literature patterns without detectors.
2. **OpenEnd semantics:** define it independently of `WinningThreat.completion`; specify whether only geometric segment endpoints qualify and how gaps/overlapping windows are handled. Until then, emit no `OpenEnd` or `HAS_OPEN_END` fact.

### Minimum validation gates

- Every relation subject/object resolves to a declared entity.
- Every fact has one provenance record and one epistemic class.
- Existing VCF and flat proof replay must pass before `CERTIFIED` facts are emitted.
- Exact-partial facts cannot claim complete optimal actions.
- Learned evidence cannot use tactical predicates such as `CREATES`, `BLOCKS`, or `SUPPORTS` unless a separate certified/derived source emits them.
- Raw and D4-transformed exports round-trip, and canonical semantic keys agree across all eight transforms.
- Export of frozen benchmark does not mutate the benchmark and reproduces 94 exact state records, 83 proof-bearing states, and 243 replay-backed proofs.

No `semantic_kg_candidate_schema.json` is emitted by this audit. The requested JSON was optional, and creating a formal schema before the two bounded `Pattern`/`OpenEnd` semantic decisions would either encode ambiguity or overclaim unsupported entities. The source-supported minimum contract above is sufficient to begin the next design step without changing current code.

## J. Source-reference index for conclusions

| Conclusion | Primary exact references |
|---|---|
| Spatial graph has four relations and stable size-local edges | `azgomoku/graph.py:3-17` |
| Board, action, player semantics | `azgomoku/game.py:4-36` |
| Exact solver reports all root action values/optimal actions | `azgomoku/solver.py:15-28`, `:75-95` |
| Threat/defense/move-threat classes | `azgomoku/tactics.py:17-62` |
| Window/immediate threat extraction | `azgomoku/tactics.py:65-113` |
| Mandatory defenses | `azgomoku/tactics.py:116-129` |
| Move threat classification | `azgomoku/tactics.py:132-196` |
| Flat tactical proofs and concepts | `azgomoku/tactics.py:199-225` |
| VCF proof/result schema | `azgomoku/vcf.py:20-74` |
| VCF OR/AND reasoning | `azgomoku/vcf.py:123-205` |
| VCF replay and reduction | `azgomoku/vcf.py:209-294` |
| Ground-truth exact/partial/unknown separation | `azgomoku/ground_truth.py:64-100` |
| H1 state/provenance schema and fail-closed validation | `azgomoku/h1_schema.py:51-67`, `:74-160` |
| Flat proof/certificate replay and frozen annotation | `investigation/e3b_common.py:73-197` |
| Frozen benchmark scope/replay counts | `diagnostic/h1_benchmark_v1/manifest.json:8-44` |
| Example tactical frozen proofs | `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl:2` |
| Example VCF certificate | `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl:19` |
| State/cell explanation identity | `azgomoku/explanation/explanation_schema.py:11-43` |
| Structural vs learned graph evidence | `azgomoku/explanation/model_evidence.py:14-41` |
| MCTS evidence semantics | `azgomoku/mcts.py:5-46`; `azgomoku/explanation/mcts_trace.py:5-17` |
| D4 transforms/canonical state key | `azgomoku/symmetry.py:10-87` |
| Cross-module coordinate/D4 gate | `investigation/e3b_graph.py:45-101` |
| Proof-edge overlap predicate | `investigation/evaluate_h1.py:44-59` |
| `knowledge.svg` is proof-flat contrast, not a KG | `azgomoku/explanation/rendering/knowledge_svg.py:1-205` |
| Batch knowledge scope is 83 rendered / 11 no-proof | `investigation/e3b5_knowledge.py:29-95` |
| Existing contract separates solver result from explanation proof | `docs/contracts/h1_correctness.md:3-9` |

## Final verdict

**PARTIALLY READY — current Level 1, with substantial but unmaterialized Level 2 ingredients.**

The repository is close in **source facts**, especially for Threat, Proof, Move, exact solver output, VCF forced responses, structural edges, and learned evidence. It is not yet close enough in **knowledge representation** to claim a Semantic KG: Pattern/OpenEnd contracts, stable semantic identities, typed relation facts, fact-level provenance, and epistemic-class serialization are still missing. The minimum path to Level 3 is a thin semantic adapter/export layer plus two narrow semantic definitions, not a project rewrite.
