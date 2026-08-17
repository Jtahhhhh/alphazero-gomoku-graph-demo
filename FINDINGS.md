# Findings

## System Map

| Component | File(s) | Role |
|---|---|---|
| Exact solver | `azgomoku/solver.py` | Full-width minimax over all legal actions; returns exact values and the full optimal-action set. |
| One-sided tactical solver | `azgomoku/vcf.py` | Proves attacker wins only, replays certificates, and returns `exact_partial` or `unknown`. |
| Ground-truth router | `azgomoku/ground_truth.py`, `investigation/generate_h1_benchmark.py`, `azgomoku/benchmark.py` | Routes exact solver first, then VCF; freezes the benchmark as exact-complete 6x6/k=4 gold. |
| Proof annotation | `investigation/e3b_common.py`, `azgomoku/semantic/extract_proofs.py`, `azgomoku/proof_replay.py` | Adds replay-verified tactical proofs and VCF certificates without changing labels. |
| Graph evidence | `azgomoku/graph.py`, `azgomoku/explanation/model_evidence.py`, `investigation/e3b_graph.py` | Builds the cell graph, extracts model attention, and renders proof-vs-graph comparisons. |
| Metrics | `azgomoku/metrics/semantic_alignment.py`, `azgomoku/metrics/attention.py`, `investigation/e3b_pipeline.py`, `investigation/evaluate_h1.py` | Computes proof alignment, baselines, and attention-collapse diagnostics. |
| Semantic KG | `azgomoku/semantic/export_kg.py`, `azgomoku/semantic/extract_tactics.py`, `azgomoku/semantic/validation.py`, `azgomoku/semantic/evidence_schema.py` | Exports and validates the replay-backed semantic knowledge graph. |

## Verdicts

| Task | Claim checked | Verdict | Evidence | Risk |
|---|---|---|---|---|
| 1 | Ground truth is offense-only / only forced win lines count | PARTIAL | `azgomoku/solver.py:75`, `azgomoku/ground_truth.py:74`, `azgomoku/ground_truth.py:86`, `azgomoku/vcf.py:299`, `azgomoku/vcf.py:334`, `investigation/e3b_common.py:46` | The frozen benchmark uses full minimax exact labels where available, but proof-bearing attention alignment is still only a tactical subset. |
| 2 | `unknown` states are dropped, causing selection bias | CONFIRMED | `azgomoku/benchmark.py:21`, `azgomoku/benchmark.py:34`, `investigation/generate_h1_benchmark.py:131`, `investigation/e3b_pipeline.py:81`, `investigation/e3b_pipeline.py:88` | Unknown/partial states do not enter the final denominator; there is no MCAR or abstain-shift audit in code. |
| 3 | Critical cells should be the union of all valid proofs, not one line | VIOLATED | `azgomoku/metrics/semantic_alignment.py:24`, `azgomoku/metrics/semantic_alignment.py:29`, `azgomoku/metrics/semantic_alignment.py:36`, `investigation/evaluate_h1.py:41`, `investigation/e3b_graph.py:150`, `investigation/semantic_xai.py:290` | Main metrics score each proof separately and average them; `evaluate_h1.py` also conditions proof choice on the model-selected action. |
| 4 | Threat vocabulary and game variant are consistent | PARTIAL | `azgomoku/game.py:20`, `azgomoku/game.py:30`, `azgomoku/benchmark.py:42`, `azgomoku/tactics.py:173`, `azgomoku/tactics.py:195`, `azgomoku/extract_tactics.py:79`, `investigation/e3b_pipeline.py:348` | Variant is consistently 6x6/k=4 with no forbidden-move logic, but the tactical vocabulary is intentionally narrow: immediate win, mandatory block, simple fork, and VCF five/four candidates only. |
| 5 | Graph representation has no ceiling for solver-critical relations | PARTIAL | `azgomoku/graph.py:3`, `azgomoku/graph.py:17`, `azgomoku/explanation/model_evidence.py:21`, `azgomoku/explanation/model_evidence.py:34`, `azgomoku/semantic/extract_proofs.py:237`, `azgomoku/metrics/semantic_xai.py:13` | Current proof types map cleanly to geometric edges and node targets, but higher-order tactics are not modeled as first-class relations. |
| 6 | Baselines are strong enough | VIOLATED | `azgomoku/metrics/semantic_alignment.py:42`, `investigation/e3b_pipeline.py:198`, `investigation/evaluate_h1.py:45`, `investigation/h3_evaluate.py:39` | Only random and structural baselines are implemented; there is no proximity/last-move or simple threat-detector baseline. |
| 7 | Metric definitions are fully clean and non-circular | PARTIAL | `azgomoku/metrics/semantic_alignment.py:15`, `azgomoku/metrics/semantic_alignment.py:24`, `azgomoku/metrics/semantic_alignment.py:29`, `azgomoku/metrics/semantic_alignment.py:33`, `investigation/evaluate_h1.py:41`, `investigation/e3b_pipeline.py:276` | The formulas are deterministic, but P@K and R@K collapse because K is set to the positive count, and `evaluate_h1.py` chooses the proof subset using the model's own action. |
| 8 | Heuristics leak into ground truth | CONFIRMED | `azgomoku/ground_truth.py:74`, `azgomoku/ground_truth.py:86`, `investigation/e3b_common.py:38`, `azgomoku/semantic/validation.py:88`, `azgomoku/semantic/extract_proofs.py:137` | Replay-verified proofs and exact solver labels are used; no distance/score/attention heuristic defines the gold labels. |
| 9 | Coverage reporting is conditional, not over-claimed | CONFIRMED | `investigation/e3b_pipeline.py:81`, `investigation/e3b_pipeline.py:276`, `investigation/e3b_pipeline.py:348`, `investigation/e3b5_knowledge.py:56`, `investigation/e3b5_knowledge.py:88` | The frozen pipeline explicitly labels mid-phase as suggestive only and excludes no-proof states from alignment claims. |

## Scope Decision

`B) Narrower than claim`

The exact solver labels are sound on the frozen 6x6/k=4 benchmark, but the attention conclusions are conditional on a narrower proof-bearing subset. The repo is honest about that in the final E-3b pipeline, yet some helper scripts still use model-conditioned proof selection and the baseline suite is thin.

## Top Risks

1. The main alignment metric is per-proof, not union-based over all valid proofs.
2. No-proof states are excluded from the alignment denominator, so coverage is conditional.
3. The baseline set is too weak to rule out simple heuristic explanations.
4. The current graph/proof vocabulary is intentionally narrow, so any future move beyond immediate wins, blocks, and simple forks will need new representation and metrics.
