# Semantic XAI reproduction pipeline

This document is the entry point for reproducing the frozen H1 → Semantic KG →
Evidence Overlay → Semantic XAI release without changing its scientific contract.

## Contract boundary

Two H1 datasets exist and are intentionally not interchangeable:

- `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl` is the immutable release
  benchmark: 94 exact states, 83 proof-bearing states, and 243 valid proofs.
- `diagnostic/h1_tactical.jsonl` is the legacy compatibility benchmark. It is
  retained for historical scripts and must not replace the frozen v1 denominator.

Candidate generation writes to `results/`; it must never overwrite either frozen
benchmark file. Unknown or partial solver outcomes remain outside the exact gold
denominator.

## Data flow and ownership

```text
Frozen H1 benchmark v1
        |
        v
Semantic KG (EXACT / CERTIFIED / DERIVED)
        |
        v
Evidence Overlay (LEARNED / search observations)
        |
        v
Semantic XAI metrics and release gates
```

Reusable code lives below `azgomoku/`. `investigation/` owns CLI argument parsing,
path resolution, orchestration, rendering, and legacy compatibility exports. No
module below `azgomoku/` may import `investigation/`.

| Layer | Canonical API | Compatibility/orchestration |
| --- | --- | --- |
| Artifact I/O | `azgomoku.artifacts` | existing import aliases |
| Benchmark routing | `azgomoku.benchmark` | `investigation.e3b_common` |
| Proof replay | `azgomoku.proof_replay` | `investigation.e3b_common` |
| Alignment/collapse | `azgomoku.metrics.*` | `evaluate_h1`, `e3b_pipeline` |
| Semantic evidence | `azgomoku.semantic.evidence_export` | `investigation.semantic_evidence_export` |
| Semantic XAI pure metrics | `azgomoku.metrics.semantic_xai` | `investigation.semantic_xai` |

Provenance source/function strings are part of the release contract. Moving an
implementation does not authorize rewriting those strings.

## Frozen inputs

The authoritative path/hash/size allowlist is
`configs/release_artifacts.json`, generated from
`docs/audits/source_cleanup/phase0_allowlist_hashes.csv` plus the endpoint evidence
manifest. It covers:

- frozen H1, Semantic KG, and learned-evidence artifacts;
- exactly 26 seed-7 checkpoint files used by the evidence lineage;
- all 13 compact Phase 5 release files;
- Phase 6 resume files, which remain ignored but hash-locked.

Do not run a reproducer with `semantic_kg/`, `semantic_evidence_v1/`, or
`results/semantic_xai/` as its output while validating a refactor. Use a new temp
directory and compare it to the frozen release.

## Reproduction commands

Run from the repository root with the project dependencies available on
`PYTHONPATH`.

```bash
export CLEANUP_TMP="${TMPDIR:-/tmp}/azgomoku-semantic-repro"
mkdir -p "$CLEANUP_TMP"
```

### 1. Test and import boundary

```bash
python -m pytest -q -p no:cacheprovider
```

The architecture regression test in `tests/test_source_boundaries.py` rejects any
`azgomoku → investigation` import and locks compatibility aliases and AP tie order.

### 2. Rebuild Semantic KG into temp

```bash
python -m azgomoku.semantic.export_kg \
  --benchmark diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl \
  --output "$CLEANUP_TMP/semantic_kg" \
  --pilot-output "$CLEANUP_TMP/semantic_kg_pilot"
```

Compare the generated JSONL/manifests with `semantic_kg/`. The source benchmark
and its manifest must retain their pre-run hashes, and the D4 gate must pass.

### 3. Rebuild evidence into temp

```bash
python -m investigation.semantic_evidence_export \
  --benchmark diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl \
  --base-kg semantic_kg \
  --rgat-run results/h3_pilot_v2/rgat/seed_7 \
  --rgcn-run results/h3_pilot_v2/rgcn/seed_7 \
  --skip-full-d4 \
  --output "$CLEANUP_TMP/semantic_evidence_v1"
```

`--skip-full-d4` is valid only because the frozen base already contains a passing
`semantic_kg/d4_release_gate.json`. Root and endpoint entities/facts/provenance,
both manifests, and `evidence_release_gate.json` must be byte-identical to the
frozen evidence overlay.

### 4. Reproduce Semantic XAI into temp

```bash
python -m investigation.semantic_xai \
  --benchmark diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl \
  --base-kg semantic_kg \
  --evidence semantic_evidence_v1 \
  --legacy-progress results/h1_integration/e3b/evaluation_progress.json \
  --legacy-developmental results/h1_integration/developmental/developmental_metrics.csv \
  --output "$CLEANUP_TMP/semantic_xai"
```

The release gate must report:

- `passed: true`;
- 94 exact states, 83 proof-bearing states, 243 proofs;
- 1,115 endpoint proof rows and 441 endpoint state-semantic rows;
- network/search absolute delta at most `1e-6`;
- alignment absolute delta at most `1e-12`.

All 13 generated Phase 5 CSV/JSON/SVG files should be byte-identical. A numeric
comparison within the declared tolerance is acceptable only when a byte difference
has an explained platform-level cause; unexplained drift fails the gate.

## Artifact storage policy

- Source, tests, configs, contracts, compact gates/manifests, and small scientific
  CSV/JSON/SVG are normal Git files.
- The six frozen evidence JSONL files and exactly 26 checkpoint lineage files use
  Git LFS through `.gitattributes`.
- `results/**` is ignored by default. `.gitignore` admits only explicit
  manifest-owned Phase 5 files and checkpoint paths.
- Phase 6 resume outputs, caches, logs, and reproduction scratch directories stay
  ignored. Their required resume hashes are recorded in the release-artifact
  manifest.

Never use `git add .` for a release. Stage explicit paths and review LFS pointers,
`git diff --cached`, and frozen hashes before committing.
