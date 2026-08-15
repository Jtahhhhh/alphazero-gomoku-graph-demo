# Current architecture map

This map describes the repository as found on 2026-08-15. It is descriptive,
not a refactor already performed.

## Scientific/data flow

```text
configs/*.yaml (JSON-compatible YAML)
        |
        v
experiments.run_h3_pilot
        |
        +--> azgomoku.training --> azgomoku.mcts --> models.RGCN / models.RGAT
        |
        v
results/h3_pilot_v2/<model>/seed_7/checkpoints
        |
        +--> H1 state generation / exact + VCF routing
        |       investigation.generate_h1_benchmark
        |       azgomoku.ground_truth
        |       azgomoku.solver / azgomoku.vcf / azgomoku.offline_solver
        |
        v
diagnostic/h1_benchmark_v1 (immutable, 94 states)
        |
        +--> azgomoku.semantic.export_kg
        |       v
        |    semantic_kg (immutable base KG)
        |
        +--> investigation.semantic_evidence_export
        |       +--> model checkpoints + MCTS
        |       v
        |    semantic_evidence_v1 (learned overlay)
        |
        +--> investigation.semantic_xai
                +--> semantic_kg
                +--> semantic_evidence_v1
                +--> legacy E-3b/developmental outputs
                v
             results/semantic_xai
             CSV / JSON / SVG / Phase 5 release gate
```

Phase 6 currently adds four parameter copies under `configs/phase6_*` and
partially generated runs under `results/h3_pilot_phase6`; those outputs do not
yet represent a completed 60-iteration release.

## Runtime and training layer

| Module | Responsibility | Used by | Main artifacts | Direct tests |
| --- | --- | --- | --- | --- |
| `azgomoku/game.py` | Board state, legality, terminal logic, features | Almost all runtime/evaluation modules | State records and tensors | `test_core`, solver/tactics/semantic tests |
| `azgomoku/graph.py` | Typed cell graph and stable structural edge records | Models, evidence, semantic extractors | Edge IDs/relations | `test_graph`, semantic identity/schema tests |
| `models/rgcn.py`, `models/rgat.py` | Policy/value graph encoders | Training, evaluation, evidence export | Checkpoint state dictionaries | `test_models`, explanation/evaluation tests |
| `azgomoku/mcts.py` | Search and root visit policy | Training, H1/evidence evaluation | Root trace and policy | collapse, explanation, game-export tests |
| `azgomoku/training.py` | Self-play, augmentation, optimizer loop | H3 pilot runner | replay samples, metrics | `test_collapse_controls` |
| `azgomoku/h3_checkpoint.py` | Immutable bundle save/load and manifest | Trainer and evaluators | `.pt`, manifest | `test_h3_infrastructure` |
| `experiments/run_h3_pilot.py` | Training orchestration | CLI | run directory | indirect infrastructure tests |

## Solver and H1 layer

| Module | Responsibility | Used by | Main artifacts | Direct tests |
| --- | --- | --- | --- | --- |
| `azgomoku/solver.py` | Bounded exact negamax oracle | Router, gates, D2c/E-3 | exact labels | solver/oracle/router tests |
| `azgomoku/tactics.py` | Geometry-certified tactical primitives/proofs | VCF, annotation, KG | flat tactical proofs | tactics/threat tests |
| `azgomoku/vcf.py` | One-sided VCF with replayable certificate and abstention | Router and audit tools | partial proof certificate | VCF/oracle/router tests |
| `azgomoku/offline_solver.py` | Stronger offline exact triage | D2c-v2/E-3a.2 | triage/calibration results | offline solver tests |
| `azgomoku/ground_truth.py` | Exact → replayed VCF → unknown routing | H1 generators and arena | routed H1 records | router tests |
| `azgomoku/h1_schema.py` | Schema-v2 writer/fail-closed reader | H1 and Semantic KG | benchmark JSONL | schema tests |
| `azgomoku/symmetry.py` | D4 transforms/canonicalization | generator and semantic identity | D4 keys/gates | generator/D4 identity tests |
| `investigation/e3b_common.py` | Gold reader, proof replay/annotation, JSONL writing | E-3b and Semantic KG | frozen benchmark/proofs | E-3b/evidence tests |
| `investigation/e3b_graph.py` | Structural baselines, coordinate gate, comparison SVG | E-3b/Phase 5/knowledge | graph gates/SVG | E-3b/explanation/semantic tests |
| `investigation/e3b_pipeline.py` | Required evaluator → graph → freeze → endpoint order | CLI, developmental/arena helpers | frozen H1, endpoint outputs | E-3b tests |

The last three rows are reusable libraries living under `investigation/`; two
are imported by `azgomoku.semantic`. That is the principal dependency inversion.

## Semantic KG and evidence layer

| Module | Responsibility | Used by | Main artifacts | Direct tests |
| --- | --- | --- | --- | --- |
| `azgomoku/semantic/schema.py` | Entity/fact/provenance contracts | All KG adapters | in-memory artifact | schema/extraction/validation tests |
| `azgomoku/semantic/identity.py` | Raw IDs and D4-canonical keys | All semantic extractors | stable IDs/lineage | identity/D4 tests |
| `azgomoku/semantic/predicates.py` | Closed base predicate vocabulary | schema/extractors | ontology contract | schema/validation tests |
| `azgomoku/semantic/epistemic.py` | EXACT/CERTIFIED/DERIVED/LEARNED classes | schemas/extractors | epistemic tags | validation/extraction tests |
| `extract_state.py`, `extract_tactics.py`, `extract_proofs.py` | Deterministic base-KG adapters | `export_kg.py` | entities/facts/provenance | extraction/full-D4 tests |
| `azgomoku/semantic/export_kg.py` | Deterministic base export + D4 gate | CLI and Phase 4/5 loaders | `semantic_kg/` | export/full-D4 tests |
| `azgomoku/semantic/evidence_schema.py` | Separate learned-overlay contract v1.1 | Phase 4/5 | in-memory overlay | evidence schema/join/provenance tests |
| `azgomoku/semantic/extract_evidence.py` | Earlier in-memory evidence adapter over base artifact | Tests; not Phase 4 exporter | in-memory facts | extraction tests |
| `investigation/semantic_evidence_export.py` | Full v1.1 overlay generation, hashing, release verification | CLI and Phase 5 | `semantic_evidence_v1/` | evidence tests |
| `investigation/semantic_xai.py` | Semantic targets, alignment, contrast, bootstrap, figures, gates | CLI | `results/semantic_xai/` | semantic XAI tests |

`extract_evidence.py` and `semantic_evidence_export.py` look similar but are not
exact duplicates: they target different artifact contracts. Consolidation must
preserve that semantic difference while sharing tensor-to-observation adapters.

## Explanation and visualization layer

| Module | Responsibility | Used by | Main artifacts | Direct tests |
| --- | --- | --- | --- | --- |
| `explanation_schema.py` | State/action document and state identity | H1/semantic/investigation | explanation JSON | game/H1 tests |
| `model_evidence.py` | One evidence-enabled model forward | explanations/evaluators | network and attention evidence | explanation tests |
| `mcts_trace.py` | Root-only MCTS trace | explanation/semantic | candidate trace | explanation tests |
| `explanation_export.py` | Decision export CLI/API | game/arena | JSON + three/four SVGs | explanation tests |
| `game_export.py` | Full-game orchestration and eval/data action selection | arena/H1 generator | game tree | game-export tests |
| `rendering/*.py` | Board, graph, decision, knowledge SVGs | exporters | SVG | explanation/E-3b tests |

## Investigation script lifecycle classification

| Script | Classification | Reason |
| --- | --- | --- |
| `semantic_evidence_export.py` | ACTIVE RELEASE ORCHESTRATOR | Creates frozen Phase 4 overlay and gate |
| `semantic_xai.py` | ACTIVE RELEASE ORCHESTRATOR | Creates Phase 5 metrics/figures/gate |
| `developmental_evaluate.py` | LEGACY REPRODUCTION / ACTIVE INPUT | Phase 5 consumes its developmental CSV |
| `e3b_pipeline.py` | LEGACY REPRODUCTION / ACTIVE INPUT | Freezes H1 and provides Phase 5 endpoint lineage |
| `e3b5_knowledge.py`, `arena_knowledge.py` | ACTIVE QUALITATIVE EXPORT | Paper/arena knowledge artifacts |
| `e3b_common.py`, `e3b_graph.py`, `evaluate_h1.py` | SHARED LIBRARY MISPLACED AS INVESTIGATION | Imported across scripts and package code |
| `generate_h1_benchmark.py`, `e3a1_measure.py`, `e3a2_expand_gold.py`, `e3a_gate.py` | LEGACY REPRODUCTION | Required to reproduce benchmark lineage |
| `d2c_measure.py`, `d2c_v2_measure.py`, `d2c_v2_gates.py`, `vcf_offline_crosscheck.py` | LEGACY REPRODUCTION | Required by D2c reports and solver decision trail |
| `h3_evaluate.py`, `validate_solver.py` | SUPPORTED LEGACY CLI | Still documented/tested |

No investigation script is approved for deletion from static evidence alone.

## Current dependency violations and fragile edges

```text
azgomoku.semantic.export_kg ---------> investigation.e3b_common
azgomoku.semantic.extract_proofs ----> investigation.e3b_common
azgomoku.semantic.extract_evidence --> investigation.evaluate_h1

investigation.arena_knowledge -------> investigation.e3b_pipeline._collapse_metrics
tests -------------------------------> several private investigation helpers
```

The first three arrows violate the intended direction (core package should not
depend on orchestration). The private-helper arrows make cleanup/renames risky.
The proposed target is to move pure benchmark/proof/metric helpers into
`azgomoku` while leaving compatibility wrappers in their historical modules so
reproduction commands and provenance labels remain stable.
