# Cleanup immutable/untouched allowlist

This allowlist is the Phase 0 boundary for Source Cleanup. Paths are repository
relative. The companion `phase0_allowlist_hashes.csv` records every current
file, byte size, and SHA-256 in the hashed groups.

## Immutable scientific releases

- `diagnostic/h1_benchmark_v1/**`
- `semantic_kg/**`
- `semantic_evidence_v1/**`
- `results/semantic_xai/**`
- `results/h1_integration/e3b/**`
- `results/h1_integration/developmental/**`

## Required checkpoint lineage

- Exactly the 26 checkpoint paths named by
  `semantic_evidence_v1/manifest.json`; do not infer this allowlist from all
  `.pt` files.

## Phase 6 resume state — untouched

- `configs/phase6_*.yaml`
- `results/h3_pilot_phase6/**`
- `results/semantic_xai_phase6/**`

## Existing user work — preserve unless explicitly named by a cleanup phase

- Every `TRACKED_MODIFIED` and `UNTRACKED_RELEVANT` entry captured in
  `worktree_before.txt`.
- Existing generated/progress/cache files under `results/` remain untouched in
  cleanup round 1 even when ignored.

## Forbidden operations for this cleanup

- `git add .`
- `git checkout .`
- `git clean -fdx`
- direct writes to any frozen directory
- silent regeneration of release artifacts
- staging any path not explicitly listed in a reviewed commit slice

Any allowlisted hash drift is a stop condition, except that explicitly modified
source/tests/docs are outside the hashed immutable groups. Phase 6 resume hashes
must remain byte-identical throughout all phases.
