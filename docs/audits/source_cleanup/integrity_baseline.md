# Integrity and test baseline

Captured on 2026-08-15 before any source cleanup. All checks were read-only.

## Frozen H1 benchmark

| File | Current SHA-256 | Recorded SHA-256 | Result |
| --- | --- | --- | --- |
| `diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl` | `9abd52ef4991489586682e881e495fcb4c2ffe00fb55dc9dee1d9008aca4ff02` | same | PASS |
| `diagnostic/h1_benchmark_v1/manifest.json` | `7498c6379f139b470cf8fd3b273e5085c28d4ab5427c134f59aa0df8fca2f8b1` | Semantic KG freeze record: same | PASS |

The manifest declares 94 exact-complete 6×6/k=4 states, 83 proof-bearing
states, 243 replayed proofs, and the immutable rule “changes require
h1_benchmark_v2”.

## Frozen Semantic KG v1

| File | Current SHA-256 | Freeze SHA-256 | Result |
| --- | --- | --- | --- |
| `semantic_kg/entities.jsonl` | `784f1cb9c883f38e8d19bb779ef1a6be1b9175743fc9bd79c7e9f9c6f4b3d1d7` | same | PASS |
| `semantic_kg/facts.jsonl` | `da793f2e66c5a953686175a22549618974cccd38712ee59e7e93e83c1284355e` | same | PASS |
| `semantic_kg/provenance.jsonl` | `00c5f2fa2aa4ead063efdbe82db43133bd1d33776a45b0f8334a406031481594` | same | PASS |
| `semantic_kg/manifest.json` | `28c2715fbc9b16a75dc7e849a907a807c5097918ff49a92fd0b76583533998a8` | same | PASS |

Counts: 32,495 entities; 78,840 facts; 1,144 provenance records. The stored
D4 release gate is PASS for 94×8 transforms and reports the base KG unchanged.

## Frozen semantic evidence overlay v1

| File | Current SHA-256 | Manifest SHA-256 | Result |
| --- | --- | --- | --- |
| `semantic_evidence_v1/entities.jsonl` | `0e98d5dc06c60681df7da191201c95da8e94e554601db6610cffa5057c1c7647` | same | PASS |
| `semantic_evidence_v1/facts.jsonl` | `191e8a78c58ae64ad5287591f0fb42b834762ec8d81d87f1d5f624eb511842fc` | same | PASS |
| `semantic_evidence_v1/provenance.jsonl` | `ad7c5bada0aa8aecb3a9882317d29faa1de0d5e2835379de02d5d95e4f2854b6` | same | PASS |
| `semantic_evidence_v1/manifest.json` | `87c0cf2e0f6e4cf5cf1ed4a0f6eb27af0115a73f61e811e17185afa0d20e738e` | downstream lineage reference | RECORDED |

Counts: 286,488 entities; 703,368 facts; 272,036 provenance records. The
evidence release gate reports `passed=true`, no learned tactical truth, and an
unchanged base KG.

## Phase 5 release

`results/semantic_xai/phase5_release_gate.json` reports `passed=true`.
Legacy reproduction compared 614 values; maximum absolute delta was
`4.209578037261963e-07`, below the `1e-6` network/search tolerance, with
alignment tolerance `1e-12` and zero reported alignment delta.

## Test baseline

Command:

```text
PYTHONPATH=.h3deps <bundled-python> -m pytest -q -p no:cacheprovider
```

Result: **113 passed, 1 failed** in 26.32 seconds.

The only failure was
`tests/test_explanation.py::test_optional_knowledge_svg_is_registered_beside_existing_three`.
The SVG is written as UTF-8, but the test calls `Path.read_text()` without an
encoding on Windows, so Python used cp1252 and raised `UnicodeDecodeError` on a
UTF-8 middle-dot sequence. This is a platform-sensitive test-read defect, not
an observed scientific metric or artifact-integrity change. It must be fixed
and the full suite made green before cleanup begins.

## Environment note

The repository `.venv` was created under WSL and points to `/usr/bin/python3.12`;
it is not directly runnable from PowerShell. The audit used the Codex bundled
Windows Python with the existing `.h3deps` packages. This split environment is
a reproducibility risk and should be documented or replaced with a lock/setup
command, but the audit did not alter either environment.
