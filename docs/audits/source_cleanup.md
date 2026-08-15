# Source Cleanup Audit

Audit date: 2026-08-15  
Branch/HEAD: `master` / `cc7d334`  
Scope: **audit only** — no source, test, config, scientific artifact, or existing
worktree change was edited, moved, deleted, staged, or committed.

## Executive verdict

The repository is scientifically coherent enough to consolidate, but it is not
safe to start deleting or moving files yet.

Three conditions block cleanup from starting immediately:

1. The baseline worktree contains 13 tracked modifications, 92 expanded
   untracked files, and 1,293 ignored result files. The untracked set includes
   the frozen H1 benchmark, Semantic KG, evidence overlay, most Phase 4–5 source,
   tests, reports, and contracts.
2. The current full test baseline is **113 passed, 1 failed**. The failure is a
   Windows default-encoding read in a test, but cleanup must begin from a green
   baseline rather than silently accepting it.
3. `.gitignore` currently ignores all of `results/`, hiding both disposable
   output and required release gates/reproduction inputs. Artifact policy must
   be made selective before any generated-file cleanup.

Frozen integrity is healthy: the H1 benchmark, all four base Semantic KG files,
all three full evidence JSONL files, the D4 gate, the evidence gate, and the
Phase 5 release gate match their recorded state. Those assets must remain
immutable during cleanup.

Inventory scope (excluding `.git`, environments, caches, `build`, `dist`, and
the audit output itself): **1,602 files, 2,553,031,024 bytes**. This includes
100 Python files, 69 checkpoints, 18 JSONL files, and 925 SVGs.

The retained audit evidence is under [`docs/audits/source_cleanup/`](source_cleanup/):

- `worktree_before.txt`
- `architecture_map.md`
- `cleanup_allowlist.md`
- `integrity_baseline.md`
- `phase0_allowlist_hashes.csv`

Mechanical inventories, the one-off generator, and temporary reproduction copies
were removed after the final cleanup gates passed; they are reproducible and are
not release inputs.

## 1. Current repository architecture

The current flow is:

```text
training configs
  -> H3 pilot training/checkpoints
  -> exact/VCF-routed H1 candidates
  -> immutable 94-state H1 benchmark
  -> frozen base Semantic KG
  -> separate learned-evidence overlay
  -> provenance-aware semantic evaluator
  -> Phase 5 CSV/JSON/SVG and release gate
```

The domain layers themselves are sensible:

| Layer | Current locations |
| --- | --- |
| Game/graph/search/training | `azgomoku/`, `models/`, `experiments/` |
| Exact and tactical truth | `solver.py`, `tactics.py`, `vcf.py`, `offline_solver.py`, `ground_truth.py` |
| H1 schema/freeze | `h1_schema.py`, `symmetry.py`, `investigation/e3*` |
| Base Semantic KG | `azgomoku/semantic/`, `semantic_kg/` |
| Learned evidence | `evidence_schema.py`, `semantic_evidence_export.py`, `semantic_evidence_v1/` |
| Semantic evaluation | `evaluate_h1.py`, `e3b_pipeline.py`, `semantic_xai.py` |
| Reports/results | root reports, `docs/`, `results/` |

The main structural defect is boundary placement: reusable benchmark, proof,
and metric code lives in `investigation/`, while package modules import it.
See `docs/audits/source_cleanup/architecture_map.md` for the complete
module → consumer → artifact → test map.

## 2. Git/worktree status

Baseline classification:

| Class | Count | Key contents | Cleanup rule |
| --- | ---: | --- | --- |
| `TRACKED_MODIFIED` | 13 | arena/H1/VCF source, tests, README, `.gitignore`, training manifest/log | Preserve; never stage as cleanup by default |
| `UNTRACKED_RELEVANT` | 92 | Phase 4–6 source/tests/config/docs plus frozen artifacts | Preserve and establish ownership before cleanup |
| `UNTRACKED_GENERATED` / ignored | 1,293 | results, checkpoints, logs, metrics, SVG | Classify by release policy before deletion |
| `UNRELATED_USER_CHANGE` | all pre-audit dirty entries | user-owned pre-existing work | No edit/stage/revert |
| `IGNORABLE_CACHE` | excluded | environments, `.pytest_cache`, `__pycache__`, `.pyc` | Safe cleanup candidate after process check |

There were no staged changes. The current branch is `master`. The complete
pre-audit status and diff summary are recorded in `worktree_before.txt`.

## 3. Core source modules

| Responsibility | Canonical implementation today | Assessment |
| --- | --- | --- |
| Board/state | `azgomoku/game.py` | Clear and heavily tested |
| Typed graph | `azgomoku/graph.py` | Clear and stable |
| Models | `models/rgcn.py`, `models/rgat.py` | Clear; do not relocate during this cleanup |
| MCTS | `azgomoku/mcts.py` | Clear; numerical behavior is protected |
| Training | `azgomoku/training.py`, `experiments/run_h3_pilot.py` | Clear, but config protocol fields span two layers |
| Checkpoint bundle | `azgomoku/h3_checkpoint.py` | Canonical; checkpoint-index validation is duplicated elsewhere |
| Exact ground truth | `solver.py`, `ground_truth.py` | Canonical routing is clear |
| Tactical/VCF proofs | `tactics.py`, `vcf.py` | Canonical algorithms are clear |
| Offline triage | `offline_solver.py` | Semantically distinct from the production exact solver; keep |
| H1 schema | `h1_schema.py` | Canonical and fail-closed |
| D4 transforms | `symmetry.py` | Canonical; frozen identity semantics must not change |
| Base KG contract | `azgomoku/semantic/{schema,identity,predicates,epistemic}.py` | Clear critical zone |
| Base KG export | `azgomoku/semantic/export_kg.py` | Canonical export, but imports proof replay from `investigation` |
| Evidence contract | `azgomoku/semantic/evidence_schema.py` | Canonical overlay schema v1.1 |
| Evidence export | `investigation/semantic_evidence_export.py` | Release-canonical behavior, wrong layer and too large |
| Semantic metrics | `investigation/evaluate_h1.py`, `e3b_pipeline.py`, `semantic_xai.py` | Reused, duplicated, and partly private |

## 4. Duplicate code candidates

### Exact AST duplicates

| Implementations | Classification | Proposed canonical owner | Risk |
| --- | --- | --- | --- |
| `export_kg._write_json` and `semantic_evidence_export._write_json` | `EXACT_DUPLICATE` | `azgomoku/artifacts.py` atomic writer | low |
| `export_kg.sha256_file` and `e3b_common.sha256_file` | `EXACT_DUPLICATE` | `azgomoku/artifacts.py` | low, but provenance/import compatibility matters |
| `developmental_evaluate._mean` and `e3b_pipeline._mean` | `EXACT_DUPLICATE` | `azgomoku/metrics/common.py` or local `mean_or_none` | low |
| `e3a1_measure._save_json` and `e3a2_expand_gold._write_json` | `EXACT_DUPLICATE` | shared artifact writer | low |
| `e3a2_expand_gold._load_records`, `evaluate_h1.load_records`, `h3_evaluate.load_records` | `EXACT_DUPLICATE` | benchmark/JSONL reader with explicit validation mode | medium |

`duplicate_functions.csv` records the machine-derived groups. Exact file-byte
duplicates are almost entirely repeated generated SVG/JSON assets: 227 of 229
groups live wholly under `results/`. They total only about 4.8 MB and are not a
valid reason to delete scientific outputs.

### Similar but semantically different

| Candidate | Classification | Required handling |
| --- | --- | --- |
| `semantic.extract_evidence.extract_evidence` vs Phase 4 exporter extraction | `SEMANTICALLY_DIFFERENT` contracts | Share low-level observation adapters only; do not merge artifact schemas |
| `evaluate_h1.average_precision` vs `semantic_xai._average_precision` | `SEMANTICALLY_DIFFERENT` tie-breaking | Preserve Phase 5 tie-ID behavior and reproduction tolerances |
| `developmental_evaluate.load_checkpoint_index` vs evidence exporter equivalent | `LEGACY_COMPATIBILITY` with different seed/schedule constraints | Parameterize validation without weakening either caller |
| `explanation_export.load_model`, D2c resized loader, bundle loader | `SEMANTICALLY_DIFFERENT` checkpoint formats | Document formats; no blind merge |
| JSON writers in several investigation scripts | mixed atomic/non-atomic semantics | Canonicalize only where byte format and crash safety are specified |
| base-KG and gold JSONL readers | different fail-closed eligibility rules | Keep explicit modes; never replace with a permissive generic loader |

The largest consolidation opportunity is not byte duplication but responsibility
duplication across the three large scripts: `semantic_evidence_export.py`
(703 lines), `semantic_xai.py` (1,007 lines), and `e3b_pipeline.py` (482 lines).
They should retain orchestration only after pure helpers are extracted.

## 5. Dead code candidates

Static AST inspection found these low-risk unused imports in non-test code:

| File | Unused binding |
| --- | --- |
| `azgomoku/offline_solver.py:11` | `GomokuState` |
| `azgomoku/vcf.py:12` | `creates_five` |
| `investigation/developmental_evaluate.py:15` | `phase_of` |
| `investigation/e3b_graph.py:8` | `RELATIONS` |
| `investigation/e3b_pipeline.py:28` | `replay_record_proofs` |
| `investigation/evaluate_h1.py:9` | `math` |
| `investigation/h3_evaluate.py:3` | `math` |
| `investigation/semantic_evidence_export.py:8` | `hashlib` |
| `investigation/semantic_xai.py:15` | `RELATIONS` |
| `investigation/validate_solver.py:6` | `numpy as np` |

Package `__init__.py` re-exports were deliberately excluded from dead-code
judgment even though a simple AST checker reports them as unused.

No `.py` file is approved for deletion. Five investigation CLIs have no direct
Python/doc importer in the current tree (`d2c_measure.py`, `d2c_v2_measure.py`,
`d2c_v2_gates.py`, `e3a_gate.py`, `vcf_offline_crosscheck.py`), but their names,
artifacts, and reports establish historical reproduction roles. Static
unreferenced status is insufficient to call them obsolete.

## 6. Legacy-but-required files

| File/tree | Why it must remain |
| --- | --- |
| `diagnostic/h1_tactical.{jsonl,summary.json}` | Legacy H1 tests and older evaluator defaults still reference it |
| `investigation/evaluate_h1.py` | Phase 5, E-3b, H3, arena, semantic extraction, and tests import its helpers |
| `investigation/e3b_common.py` | Frozen proof replay/annotation lineage is embedded in Semantic KG provenance |
| `investigation/e3b_graph.py` | Structural baseline and D4 coordinate gate used by Phase 5 and renderers |
| `investigation/e3a*.py` | Reproduces the candidate → calibration → gold → freeze lineage |
| `investigation/d2c*.py`, `vcf_offline_crosscheck.py` | Reproduces solver-go/no-go reports and fail-closed conclusions |
| `semantic_kg/pilot/` | Historical pilot gate preceding full freeze |
| `results/h1_integration/e3b/` | P7/Phase 5 legacy comparison input and graph gates |
| `results/h1_integration/developmental/` | Phase 5 developmental comparison input |
| iter-60 RGAT/RGCN seed-7 checkpoints | Evidence/Phase 5 lineage and P7 reproduction |

These may later move under clearly named legacy/reproduction directories, but
compatibility entry points and provenance strings must survive the move.

## 7. Generated artifacts

| Class | Examples | Policy |
| --- | --- | --- |
| Scientific release artifact | frozen H1, `semantic_kg/`, evidence manifests/gates, Phase 5 gate/CSV/SVG | KEEP; immutable; track through Git/LFS or release storage |
| Reproducible generated output | E-3a calibration, arena exports, intermediate developmental files | REVIEW; keep only manifests/compact evidence required by reports |
| Active/incomplete experiment | `results/h3_pilot_phase6/`, `results/semantic_xai_phase6/` | KEEP UNTOUCHED; do not call release-ready |
| Large checkpoint | 69 `.pt` files, about 1.75 GB | Keep required lineage checkpoints; ignore or LFS by explicit manifest policy |
| Temporary/cache | `.cache.pt`, progress JSON, local logs, raster previews | IGNORE after confirming no active resume depends on them |
| Cache/environment | `.venv`, `.h3deps`, `__pycache__`, `.pytest_cache` | IGNORE; locally removable |

Three evidence JSONL files are approximately 179, 210, and 239 MiB. The remote
is GitHub, so they cannot be added as ordinary Git blobs under GitHub's normal
100 MiB limit. Git LFS is installed. Before tracking, choose one explicit policy:

- Git LFS for immutable JSONL/checkpoints plus normal-Git manifests and gates; or
- external versioned release storage, with hashes/URLs in normal-Git manifests.

The current blanket `results/` ignore is unsafe because it also hides six
untracked gate/manifest files and the ongoing Phase 6 tree.

## 8. Naming inconsistencies

| Current pattern | Problem | Proposed rule |
| --- | --- | --- |
| `D2c_REPORT`, `E3a_REPORT`, `E3b_REPORT`, `developmental_REPORT` | Mixed case/phase syntax at repo root | Keep historical filenames initially; future `docs/reports/phase_<id>_<topic>.md` aliases |
| `evaluate_h1.py` vs `h3_evaluate.py` vs `semantic_xai.py` | Verb/object order varies | Canonical CLI aliases: `evaluate_h1`, `evaluate_h3`, `evaluate_semantic_xai` |
| `semantic_evidence_export.py` vs `export_kg.py` | Verb position varies | Canonical API under `azgomoku.semantic`; thin CLI wrappers retain old commands |
| `h3_pilot_v2`, `h3_pilot_phase6`, `semantic_xai_phase6` | Version and phase concepts mixed | Use run IDs from manifests/configs, not directory-name inference |
| `schema_version` values `1`, `"1.0"`, `"1.1"`, `2` | Multiple independent schemas can look inconsistent | Keep values; document artifact type with every version |

Ontology entity types, predicates, JSON keys, IDs, and frozen directory names
must not be renamed. Python is already predominantly `snake_case`/`PascalCase`.

## 9. Config and path inconsistencies

- No executable Python/config file contains the audited repository's absolute
  `D:/...` or `/mnt/d/...` path. The only absolute workspace path is a WSL setup
  example in `README.md`; it should become a generic `cd <repo>` example.
- CLI defaults are repo-relative but assume execution from repository root.
  A shared `repo_root()` helper is optional; explicit CLI paths are safer for
  release commands.
- `evaluate_h1.py`, `h3_evaluate.py`, and `e3a_gate.py` default to the legacy
  `diagnostic/h1_tactical.jsonl`, while Phase 4–6 uses
  `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl`. This difference must be
  named `legacy` rather than silently unified.
- The four Phase 6 config files are near-copies differing primarily by model,
  seed, and run ID. Separate immutable run configs are legitimate, but they
  should be generated/validated from one protocol schema to prevent drift.
- `run_h3_pilot.py` consumes the training fields but does not act on
  `h1_eval_every` or `mcts_eval_iterations`; those values remain checkpoint
  provenance/protocol metadata. They must not be deleted merely because the
  trainer does not branch on them.
- Project requirements are minimum ranges only; there is no locked environment.
  The WSL `.venv` and Windows `.h3deps` split already produced a platform-only
  test failure. Add a reproducible environment record before release cleanup.

Hard-coded values such as 6×6/k=4, seed 7, iter 60, 50 playouts, and Phase 5
bootstrap settings belong to different categories. Benchmark properties and
legacy reproduction constants must remain fixed; Phase 6 run parameters belong
in validated run configs; library defaults should remain conservative.

## 10. Documentation inconsistencies

1. `README.md` first documents the frozen v1 benchmark, then later calls
   `diagnostic/h1_tactical.jsonl` the “fixed benchmark”. These are distinct
   legacy/current assets.
2. The later README regeneration command calls
   `investigation.generate_h1_benchmark` without the now-required
   `--checkpoint`; as written, it fails argument parsing.
3. README's final source tree omits `azgomoku/semantic`, `semantic_kg`,
   `semantic_evidence_v1`, and the Phase 5 evaluator/results.
4. README has no canonical Phase 4 export/validation or Phase 5 reproduction
   command, despite those being the current release-critical paths.
5. Historical reports record 50, 54, 76, 80, 86, and 114 passing tests. These
   are chronological snapshots, not necessarily contradictions, but each must
   be dated/labeled so readers do not treat older counts as current.
6. Phase reports and audits occupy the repository root, while contracts occupy
   `docs/`. A single `docs/guides/semantic_xai_reproduction.md` index is missing.
7. The current independently run baseline is 113 pass/1 fail, so the Phase 4/5
   reports' `114 passed` statement is historical until the encoding-sensitive
   test is fixed and rerun.

## 11. Dependency risks

| Risk | Evidence | Severity | Required mitigation |
| --- | --- | --- | --- |
| Core package imports investigation | `semantic/export_kg.py`, `extract_proofs.py`, `extract_evidence.py` | high | Move pure helpers into `azgomoku`; retain wrappers |
| Release-critical source/artifacts untracked | 92 untracked files include all Semantic KG additions | critical | Establish an intentional baseline commit/LFS release before cleanup |
| Blanket `results/` ignore | 1,293 files hidden, including gates | high | Selective artifact policy |
| Provenance embeds file/function paths | KG JSONL names `investigation.e3b_common`, source functions, checkpoint paths | critical | Preserve lineage strings or version regenerated artifacts; never silently change |
| Private helper imports | `_collapse_metrics` imported across scripts/tests | medium | Promote public canonical API before moving callers |
| Large GitHub-incompatible blobs | three evidence files exceed 100 MiB | high | Git LFS or external release storage |
| Baseline test failure | cp1252 read of UTF-8 SVG | medium | Make test explicit UTF-8; rerun 114/114 |
| Phase 6 partial output | only RGAT iter 0; RGCN through iter 5 in current manifests | high | Do not clean/move while a resume may depend on it |
| Duplicate metric definitions | AP/tie and alignment variants | high | Golden numeric parity tests before extraction |
| Environment not locked | WSL venv plus Windows dependency directory | medium | Add environment lock/version report |

## 12. Files that must remain immutable

### Frozen H1

- `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl`
- `diagnostic/h1_benchmark_v1/manifest.json`

### Frozen Semantic KG base

- `semantic_kg/entities.jsonl`
- `semantic_kg/facts.jsonl`
- `semantic_kg/provenance.jsonl`
- `semantic_kg/manifest.json`
- `semantic_kg/base_freeze.json`
- `semantic_kg/d4_release_gate.json`

### Phase 4 learned evidence release

- `semantic_evidence_v1/entities.jsonl`
- `semantic_evidence_v1/facts.jsonl`
- `semantic_evidence_v1/provenance.jsonl`
- `semantic_evidence_v1/manifest.json`
- `semantic_evidence_v1/evidence_release_gate.json`
- `semantic_evidence_v1/endpoint/*`

### Phase 5 release and reproduction inputs

- `results/semantic_xai/phase5_release_gate.json` and its referenced CSV/JSON/SVG
- `results/h1_integration/e3b/` gate/progress/endpoint artifacts
- `results/h1_integration/developmental/` metrics/gates
- the 26 seed-7 checkpoints listed in the evidence manifest

Exact current hashes are recorded in `docs/audits/source_cleanup/integrity_baseline.md`.
Cleanup must compare hashes before and after; any difference is a stop condition.

## 13. Proposed target tree

This is a minimal consolidation target, not a request to rewrite the architecture:

```text
azgomoku/
  artifacts.py                 # hashing + byte-stable atomic JSON/JSONL I/O
  benchmark.py                 # fail-closed gold loading and phase classification
  proof_replay.py              # replay_flat_proof / replay_record_proofs
  h3_checkpoint.py             # existing bundles + parameterized checkpoint index
  metrics/
    __init__.py
    common.py                  # entropy/mean helpers with explicit contracts
    semantic_alignment.py      # critical_ids/alignment/baselines
    attention.py               # public attention-collapse metrics
    semantic_xai.py            # pure Phase 5 target/evaluation/bootstrap functions
  semantic/
    ...                        # existing frozen ontology/schema modules unchanged
    evidence_export.py         # overlay writer/extraction/release verification API

investigation/
  generate_h1_benchmark.py     # orchestration only
  e3b_pipeline.py              # orchestration + compatibility exports
  developmental_evaluate.py    # orchestration only
  semantic_evidence_export.py  # thin compatibility CLI
  semantic_xai.py              # thin compatibility CLI
  ... legacy reproduction scripts retained

scripts/
  reproduce_semantic_xai.py    # validates by default; regeneration explicit

docs/
  guides/semantic_xai_reproduction.md  # architecture/command index, links to reports
  contracts/                   # optional later move with compatibility links
  reports/                     # optional later move of historical reports
```

Investigation scripts remain executable. Compatibility re-exports in historical
modules prevent a single large import migration and preserve recorded lineage.

## 14. Exact proposed moves, renames, and deletes

### First structural cleanup commit

| Current item | Exact proposed action | Compatibility requirement | Risk |
| --- | --- | --- | --- |
| two `sha256_file` + two exact atomic `_write_json` copies | Add `azgomoku/artifacts.py`; import/re-export old names | Byte formatting and historical imports unchanged | low |
| `e3b_common.phase_of`, `load_gold_fail_closed` | Move implementation to `azgomoku/benchmark.py`; re-export from `e3b_common` | Same exceptions/order/normalization | medium |
| `e3b_common.replay_flat_proof`, `replay_record_proofs` | Move implementation to `azgomoku/proof_replay.py`; re-export | Preserve provenance labels in exporters | high |
| pure helpers in `evaluate_h1.py` | Move `entropy`, `average_precision`, `critical_ids`, `score_alignment`, `aggregate_proofs`, `baselines` to `azgomoku/metrics/semantic_alignment.py` | Golden P7/Phase 5 parity tests | high |
| `e3b_pipeline._collapse_metrics` | Move to public `azgomoku/metrics/attention.py`; keep alias | Exact float parity | high |
| `semantic_evidence_export.OverlayWriter` and overlay release API | Move implementation to `azgomoku/semantic/evidence_export.py`; leave thin CLI | Exact JSONL order/hashes when regenerated in compatibility mode | high |
| pure Phase 5 target/alignment/bootstrap functions | Move incrementally to `azgomoku/metrics/semantic_xai.py`; leave CLI | Each extraction gated against released CSV/JSON | high |
| duplicated checkpoint-index functions | Add parameterized validator to `h3_checkpoint.py` | Callers retain exact schedule/seed checks | medium |

### Low-risk deletions inside files

Delete only the ten unused imports listed in section 5 after a green test
baseline. No source file deletion is proposed in the first cleanup.

### Filesystem deletions

After confirming no active process/resume dependency, delete only:

- `**/__pycache__/`
- `*.pyc`
- `.pytest_cache/`

Do **not** delete `.venv`, `.h3deps`, progress/cache artifacts, checkpoints,
historical scripts, or any result tree as part of the first cleanup.

### Naming/doc moves (separate commit, after source parity)

Add `docs/guides/semantic_xai_reproduction.md` first. Then, only if all internal links are
updated atomically, move historical reports into `docs/reports/` with consistent
lowercase names. The initial candidates are:

```text
docs/reports/phase_d2c.md                         -> docs/reports/phase_d2c.md
docs/reports/phase_d2c_v2.md                      -> docs/reports/phase_d2c_v2.md
docs/reports/phase_e3a.md                         -> docs/reports/phase_e3a.md
docs/reports/phase_e3a1.md                        -> docs/reports/phase_e3a1.md
docs/reports/phase_e3a2.md                        -> docs/reports/phase_e3a2.md
docs/reports/phase_e3b.md                         -> docs/reports/phase_e3b.md
docs/reports/developmental_evaluation.md               -> docs/reports/developmental_evaluation.md
docs/reports/semantic_kg_phase_1_3.md        -> docs/reports/semantic_kg_phase1_3.md
docs/reports/semantic_kg_phase_4.md          -> docs/reports/semantic_kg_phase4.md
docs/reports/semantic_kg_phase_5_experiment.md -> docs/reports/semantic_kg_phase5.md
```

These moves are optional for cleanup and lower priority than dependency
consolidation. Git history should record them as renames, not delete/recreate.

## 15. Regression tests required

### Precondition

1. Fix only the Windows test encoding read, without changing SVG bytes.
2. Run the full suite and establish **114/114 passing** (or more); no silent test
   count reduction.

### After every structural extraction

- Full `pytest -q -p no:cacheprovider`.
- Import-boundary test: no `azgomoku/**` module imports `investigation/**`.
- Golden tests for AP tie-breaking, multi-proof aggregation, baselines,
  `_collapse_metrics`, and checkpoint schedule validation.
- CLI smoke tests for all compatibility module names.

### Release gates before commit

| Gate | Required result |
| --- | --- |
| Frozen H1 benchmark hashes | exact match |
| Four base Semantic KG hashes | exact match |
| Full Semantic KG D4 release validator | PASS; 94×8 scope |
| Evidence overlay hashes/gate | exact match/PASS if exporter untouched; full regenerate if changed |
| P7 legacy reproduction | network/search ≤ `1e-6`; alignment ≤ `1e-12` |
| Phase 5 release gate | PASS; 94 exact, 83 proof-bearing, 243 proofs |
| Scientific CSV/JSON comparison | exact or documented numeric tolerance with zero unexplained drift |
| Phase 6 | do not overwrite/infer completion; validate only after runs finish |

If any scientific metric or frozen hash changes, stop, identify the cause, and
do not accept the change as a clean refactor.

## 16. Proposed commit split

Precondition: first resolve ownership of the existing dirty/untracked Phase 4–6
work. Cleanup commits must not absorb those changes accidentally.

1. `refactor: consolidate semantic XAI shared infrastructure`
   - shared artifact, benchmark, proof, metric modules;
   - compatibility imports/wrappers;
   - no report/config/artifact changes;
   - exact parity tests.
2. `chore: standardize semantic XAI paths, configs, and documentation`
   - selective `.gitignore`/LFS policy;
   - README corrections and `docs/guides/semantic_xai_reproduction.md`;
   - optional report renames;
   - canonical reproduction commands.
3. `test: lock semantic XAI reproduction and release gates`
   - import-boundary and compatibility tests;
   - frozen hash/gate validation;
   - cleanup report and final diff inventory.

If the structural diff proves small after extraction, commits 1 and 3 may be
combined. The documentation/artifact-policy commit should remain separate
because it has different review and storage implications.

No commit or staging action is part of this audit turn.

## 17. Risk assessment

| Area | Risk | Likelihood | Impact | Decision |
| --- | --- | --- | --- | --- |
| Frozen H1/KG/evidence bytes | critical | low if untouched | invalidates release lineage | immutable hash gate |
| Metric extraction | high | medium | changes scientific claims | golden P7/Phase 5 comparison |
| Provenance path changes | critical | high during moves | regenerated hashes differ | compatibility wrappers/fixed lineage labels |
| Existing dirty worktree | critical | high | accidental mixing/loss | isolate and stage explicit paths only |
| Artifact ignore/storage | high | high | release files lost or unpushable | selective ignore + LFS/external policy |
| Legacy script deletion | high | medium | reproduction chain breaks | no file deletion in first cleanup |
| Config consolidation | medium | medium | protocol drift | schema validation, preserve run configs |
| Windows/Linux environment split | medium | high | platform-only failures | explicit UTF-8 and locked environment |
| Documentation moves | low/medium | high | broken links/commands | separate atomic docs commit |
| Cache deletion | low | low | lost local convenience only | verify no active run, then delete caches |

## File decision table

| File | Current role | Problem | Proposed action | Risk |
| --- | --- | --- | --- | --- |
| `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl` | frozen benchmark | untracked critical artifact | KEEP immutable | critical |
| `diagnostic/h1_benchmark_v1/manifest.json` | freeze contract | untracked | KEEP immutable | critical |
| `semantic_kg/entities.jsonl` | frozen KG | untracked release data | KEEP immutable | critical |
| `semantic_kg/facts.jsonl` | frozen KG | untracked release data | KEEP immutable | critical |
| `semantic_kg/provenance.jsonl` | frozen lineage | embeds source paths | KEEP immutable | critical |
| `semantic_kg/manifest.json` | KG manifest | untracked | KEEP immutable | critical |
| `semantic_evidence_v1/entities.jsonl` | learned overlay | 179 MiB, untracked | KEEP; LFS/external storage | critical |
| `semantic_evidence_v1/facts.jsonl` | learned overlay | 210 MiB, untracked | KEEP; LFS/external storage | critical |
| `semantic_evidence_v1/provenance.jsonl` | learned lineage | 239 MiB, untracked | KEEP; LFS/external storage | critical |
| `results/semantic_xai/phase5_release_gate.json` | Phase 5 gate | hidden by `results/` ignore | KEEP/TRACK compact release | critical |
| `azgomoku/semantic/export_kg.py` | base exporter | imports investigation; duplicate I/O/hash | EXTRACT shared helpers | high |
| `azgomoku/semantic/extract_proofs.py` | proof adapter | imports investigation replay | MOVE dependency to core helper | high |
| `azgomoku/semantic/extract_evidence.py` | legacy in-memory adapter | imports metric from investigation | MOVE dependency to core metric | high |
| `azgomoku/semantic/evidence_schema.py` | evidence schema | critical/frozen contract | KEEP, minimal edits | critical |
| `investigation/semantic_evidence_export.py` | Phase 4 orchestrator | 703 lines, mixed implementation/CLI | EXTRACT API; keep wrapper | high |
| `investigation/semantic_xai.py` | Phase 5 evaluator | 1,007 lines, mixed metrics/I/O/figures | EXTRACT incrementally | high |
| `investigation/e3b_common.py` | benchmark/proof library | wrong layer; duplicate hash/I/O | MOVE implementation + re-export | high |
| `investigation/e3b_graph.py` | graph/coordinate library | wrong layer; reused broadly | KEEP first; extract after metrics | medium/high |
| `investigation/e3b_pipeline.py` | P7 orchestrator | exports private metric helper | PUBLIC metric extraction | high |
| `investigation/evaluate_h1.py` | legacy evaluator/shared metrics | package imports investigation | EXTRACT metrics; keep CLI | high |
| `investigation/d2c_measure.py` | historical measurement | no direct importer | KEEP legacy reproduction | medium |
| `investigation/d2c_v2_measure.py` | historical calibration | name resembles duplicate | KEEP; semantically distinct | medium |
| `investigation/d2c_v2_gates.py` | soundness gates | no direct importer | KEEP legacy gate | medium |
| `investigation/vcf_offline_crosscheck.py` | solver crosscheck | no direct importer | KEEP legacy lineage | medium |
| `results/h1_integration/e3b/` | P7 artifacts | ignored/generated but required | KEEP; selective tracking policy | high |
| `results/h1_integration/developmental/` | Phase 5 legacy input | ignored/generated but required | KEEP; selective tracking policy | high |
| `results/h3_pilot_phase6/` | incomplete Phase 6 runs | blanket-ignored; resume state | KEEP untouched | high |
| `.gitignore` | artifact policy | blanket `results/` hides gates | REVIEW/replace selectively | high |
| `README.md` | entry documentation | stale benchmark/invalid command; no Phase 4–5 entry | CORRECT after source parity | medium |
| `tests/test_explanation.py` | regression test | default Windows decoding | explicit UTF-8 read | low |
| audit inventory generator and generated CSV/tree files | reproducible audit tooling | cleanup complete | REMOVE after retaining baseline hashes/report | low |

## Final recommendation

Approve cleanup only after reviewing this plan and establishing a green,
intentional Phase 4–6 baseline. The safe order is:

```text
review audit
  -> resolve existing worktree ownership/storage
  -> make baseline tests green
  -> extract shared helpers with compatibility wrappers
  -> run hash/numeric gates after each extraction
  -> standardize artifact policy and docs
  -> final diff review
  -> stage only explicitly related files
  -> commit in logical slices
```

There is currently no evidence supporting deletion of any source script. The
highest-value cleanup is dependency-direction repair and helper consolidation,
not file-count reduction.
