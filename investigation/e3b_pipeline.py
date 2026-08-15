"""Run Phase E-3b in the required evaluator -> graph -> freeze -> evaluation order."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.h1_schema import state_from_record, validate_record
from azgomoku.h3_checkpoint import model_from_bundle
from azgomoku.mcts import search
from azgomoku.metrics.attention import collapse_metrics as _collapse_metrics
from azgomoku.metrics.semantic_alignment import aggregate_proofs, baselines, entropy
from azgomoku.solver import solve_actions
from investigation.e3b_common import (
    ANNOTATION_VERSION,
    N_MIN,
    annotate_gold,
    load_gold_fail_closed,
    phase_of,
    sha256_file,
    write_jsonl,
)
from investigation.e3b_graph import (
    coordinate_gate,
    render_comparison_svg,
    runtime_mcts_action_gate,
    structural_edges,
)
from models.rgat import RGAT
from models.rgcn import RGCN


MODELS = {"rgat": RGAT, "rgcn": RGCN}
METRIC_FIELDS = (
    "policy_top1_correct",
    "policy_optimal_mass",
    "policy_entropy",
    "value_error",
    "mcts_top1_correct",
    "mcts_optimal_mass",
    "search_gain",
    "graph_critical_mass",
    "graph_precision_at_k",
    "graph_recall_at_k",
    "graph_auprc",
    "random_critical_mass",
    "structural_critical_mass",
    "random_precision_at_k",
    "random_recall_at_k",
    "random_auprc",
    "structural_precision_at_k",
    "structural_recall_at_k",
    "structural_auprc",
    "alignment_minus_random",
    "alignment_minus_structural",
    "attention_normalized_entropy",
    "attention_structural_mae",
    "attention_head_diversity",
    "attention_topology_correlation",
    "attention_collapse_flag",
)


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def evaluator_contract_gate(records: list[dict]) -> dict:
    counts = {phase: sum(phase_of(record) == phase for record in records) for phase in ("mid", "late")}
    if len(records) != 94 or counts != {"mid": 23, "late": 71}:
        raise RuntimeError(f"unexpected E-3b gold distribution: n={len(records)}, phases={counts}")
    for record in records:
        validation = validate_record(record)
        if not validation.accepted or not validation.eligible:
            raise RuntimeError(f"evaluator rejected gold: {record['state_id']}")
        if record["solver"]["status"] != "exact_complete":
            raise RuntimeError("partial/unknown entered gold denominator")
    return {
        "passed": True,
        "fail_closed_validator": True,
        "gold_only": True,
        "unknown_in_denominator": 0,
        "partial_in_denominator": 0,
        "phase_counts": counts,
        "N_min": N_MIN,
        "phase_claims": {
            "late": "main finding eligible, n=71",
            "mid": "suggestive only; n=23 < N_min=30",
        },
        "alignment_roles": {
            "rgat": "learned-attention alignment subject",
            "rgcn": "structural baseline by design; not an alignment finding",
        },
    }


def exact_freeze_gate(records: list[dict], cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    for index, record in enumerate(records):
        state_id = record["state_id"]
        if cache.get(state_id, {}).get("matched"):
            continue
        state = state_from_record(record)
        expected = record["solver"]
        result = solve_actions(state, deadline_ms=30_000, node_budget=5_000_000)
        matched = (
            result.status == "exact"
            and result.value == expected["value"]
            and list(result.optimal_actions) == expected["optimal_actions"]
        )
        cache[state_id] = {
            "matched": matched,
            "actual_status": result.status,
            "actual_value": result.value,
            "actual_optimal_actions": list(result.optimal_actions),
            "expected_value": expected["value"],
            "expected_optimal_actions": expected["optimal_actions"],
            "nodes": result.nodes,
            "elapsed_ms": result.elapsed_ms,
        }
        write_json(cache_path, cache)
        print(json.dumps({"stage": "freeze_exact_gate", "index": index + 1, "total": len(records), "matched": matched}), flush=True)
        if not matched:
            raise RuntimeError(f"exact freeze agreement failed: {state_id}")
    return {
        "passed": all(item["matched"] for item in cache.values()) and len(cache) == len(records),
        "checks": len(records),
        "mismatches": sum(not cache[record["state_id"]]["matched"] for record in records),
        "budget_ms": 30_000,
        "node_cap": 5_000_000,
    }


def evaluate_record(record, model, model_type: str, checkpoint: Path, checkpoint_hash: str, benchmark_hash: str, playouts: int) -> dict:
    state = state_from_record(record)
    device = next(model.parameters()).device
    legal = list(map(int, state.legal_actions()))
    optimal = set(map(int, record["solver"]["optimal_actions"]))
    x = torch.from_numpy(state.features()).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, value = model(x, return_evidence=False)
    mask = torch.full_like(logits, -torch.inf)
    mask[0, legal] = logits[0, legal]
    policy = torch.softmax(mask, dim=-1)[0].cpu().numpy()
    policy_action = int(policy.argmax())

    pi, _ = search(model, state, playouts=playouts, temperature=1.0, return_root=True)
    mcts_action = int(pi.argmax())
    phase = phase_of(record)
    proofs = record.get("valid_proofs", [])
    row = {
        "benchmark_sha256": benchmark_hash,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(checkpoint),
        "iteration": 60,
        "model_type": model_type,
        "alignment_role": "learned_attention_subject" if model_type == "rgat" else "structural_baseline_by_design_not_finding",
        "state_id": record["state_id"],
        "phase": phase,
        "phase_n": 23 if phase == "mid" else 71,
        "phase_claim": "suggestive_no_conclusion" if phase == "mid" else "main_finding_eligible",
        "proof_available": int(bool(proofs)),
        "proof_count": len(proofs),
        "policy_top1_correct": int(policy_action in optimal),
        "policy_optimal_mass": float(sum(policy[action] for action in optimal)),
        "policy_entropy": entropy(policy[legal]),
        "value_prediction": float(value.item()),
        "solver_value": int(record["solver"]["value"]),
        "value_error": abs(float(value.item()) - int(record["solver"]["value"])),
        "mcts_playouts": playouts,
        "mcts_top1_correct": int(mcts_action in optimal),
        "mcts_optimal_mass": float(sum(pi[action] for action in optimal)),
        "search_gain": float(sum(pi[action] for action in optimal) - sum(policy[action] for action in optimal)),
    }

    if model_type == "rgat":
        evidence = collect_model_evidence(state, model, policy_action)
        edges = evidence["graph_evidence"]["edges"]
        scores = [float(edge["attention"]) for edge in edges]
        row.update(_collapse_metrics(edges))
    else:
        edges = structural_edges(state.size)
        scores = [float(edge["attention"]) for edge in edges]

    if proofs:
        alignment = aggregate_proofs(edges, scores, proofs)["mean"]
        random_base, structural_base = baselines(edges, proofs, record["state_id"])
        row.update({
            "graph_critical_mass": alignment["mass"],
            "graph_precision_at_k": alignment["precision_at_k"],
            "graph_recall_at_k": alignment["recall_at_k"],
            "graph_auprc": alignment["auprc"],
            "random_critical_mass": random_base["mass"],
            "structural_critical_mass": structural_base["mass"],
            "random_precision_at_k": random_base["precision_at_k"],
            "random_recall_at_k": random_base["recall_at_k"],
            "random_auprc": random_base["auprc"],
            "structural_precision_at_k": structural_base["precision_at_k"],
            "structural_recall_at_k": structural_base["recall_at_k"],
            "structural_auprc": structural_base["auprc"],
            "alignment_minus_random": alignment["mass"] - random_base["mass"],
            "alignment_minus_structural": alignment["mass"] - structural_base["mass"],
        })
    else:
        row.update({key: None for key in (
            "graph_critical_mass", "graph_precision_at_k", "graph_recall_at_k", "graph_auprc",
            "random_critical_mass", "structural_critical_mass", "alignment_minus_random", "alignment_minus_structural",
            "random_precision_at_k", "random_recall_at_k", "random_auprc",
            "structural_precision_at_k", "structural_recall_at_k", "structural_auprc",
        )})
    for key in METRIC_FIELDS:
        row.setdefault(key, None)
    return row


def _mean(rows: list[dict], key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def _baseline_columns(record: dict) -> dict:
    proofs = record.get("valid_proofs", [])
    if not proofs:
        return {key: None for key in (
            "random_precision_at_k", "random_recall_at_k", "random_auprc",
            "structural_precision_at_k", "structural_recall_at_k", "structural_auprc",
        )}
    state = state_from_record(record)
    edges = structural_edges(state.size)
    random_base, structural_base = baselines(edges, proofs, record["state_id"])
    return {
        "random_precision_at_k": random_base["precision_at_k"],
        "random_recall_at_k": random_base["recall_at_k"],
        "random_auprc": random_base["auprc"],
        "structural_precision_at_k": structural_base["precision_at_k"],
        "structural_recall_at_k": structural_base["recall_at_k"],
        "structural_auprc": structural_base["auprc"],
    }


def summarize(rows: list[dict], benchmark_hash: str) -> dict:
    summary = {"benchmark_sha256": benchmark_hash, "groups": {}}
    for model_type in ("rgcn", "rgat"):
        for phase in ("mid", "late"):
            group = [row for row in rows if row["model_type"] == model_type and row["phase"] == phase]
            key = f"{model_type}:{phase}"
            summary["groups"][key] = {
                "n": len(group),
                "claim": "suggestive; no conclusion" if phase == "mid" else "main finding eligible",
                "alignment_role": group[0]["alignment_role"],
                "alignment_n": sum(row["proof_available"] for row in group),
                "means": {metric: _mean(group, metric) for metric in METRIC_FIELDS},
            }
    rgat_groups = [value for key, value in summary["groups"].items() if key.startswith("rgat:")]
    collapse_rows = [row for row in rows if row["model_type"] == "rgat"]
    collapse_rate = _mean(collapse_rows, "attention_collapse_flag")
    summary["collapse_diagnostic"] = {
        "operational_definition": "normalized entropy >=0.98, structural MAE <=0.02, head diversity <=0.02",
        "state_collapse_rate": collapse_rate,
        "mean_normalized_entropy": _mean(collapse_rows, "attention_normalized_entropy"),
        "mean_structural_mae": _mean(collapse_rows, "attention_structural_mae"),
        "mean_head_diversity": _mean(collapse_rows, "attention_head_diversity"),
        "interpretation": "collapse is a separate diagnostic; alignment deltas are correlational",
    }
    summary["scope_notes"] = [
        "Policy/value/MCTS use all exact-complete gold in each phase.",
        "Alignment uses only replay-proof-bearing exact gold; missing proof is excluded, never scored as zero.",
        "R-GCN alignment equals the structural reference by design and is not a finding.",
        "Mid n=23 is suggestive and cannot support a strong conclusion.",
    ]
    return summary


def evaluate_models(benchmark: Path, manifest_path: Path, checkpoints: dict[str, Path], output_dir: Path, playouts: int) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_hash = sha256_file(benchmark)
    if benchmark_hash != manifest["benchmark_sha256"]:
        raise RuntimeError("benchmark hash does not match manifest")
    records = load_gold_fail_closed(benchmark)
    progress_path = output_dir / "evaluation_progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    for model_type in ("rgcn", "rgat"):
        checkpoint = checkpoints[model_type]
        model, bundle = model_from_bundle(checkpoint, MODELS)
        checkpoint_hash = sha256_file(checkpoint)
        if bundle["training_state"]["iteration"] != 60 or bundle["model_type"] != model_type:
            raise RuntimeError(f"wrong endpoint checkpoint: {checkpoint}")
        for index, record in enumerate(records):
            key = f"{model_type}:{record['state_id']}"
            if key in progress:
                if progress[key]["benchmark_sha256"] != benchmark_hash:
                    raise RuntimeError("stale evaluation progress benchmark hash")
                if "random_auprc" not in progress[key]:
                    progress[key].update(_baseline_columns(record))
                continue
            started = time.perf_counter()
            progress[key] = evaluate_record(record, model, model_type, checkpoint, checkpoint_hash, benchmark_hash, playouts)
            progress[key]["wall_seconds"] = time.perf_counter() - started
            write_json(progress_path, progress)
            print(json.dumps({"stage": "evaluate", "model": model_type, "state": index + 1, "total": len(records), "phase": phase_of(record)}), flush=True)
        write_json(progress_path, progress)
    rows = list(progress.values())
    csv_path = output_dir / "endpoint_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows, benchmark_hash)
    write_json(output_dir / "endpoint_summary.json", summary)
    return rows, summary


def freeze_benchmark(records: list[dict], source: Path, freeze_dir: Path, evaluator_gate: dict, graph_gate: dict, exact_gate: dict, annotation_stats: dict) -> tuple[Path, Path, dict]:
    if not evaluator_gate["passed"] or not graph_gate["passed"] or not exact_gate["passed"]:
        raise RuntimeError("cannot freeze before evaluator/graph/exact gates pass")
    freeze_dir.mkdir(parents=True, exist_ok=True)
    benchmark = freeze_dir / "h1_benchmark_v1.jsonl"
    manifest_path = freeze_dir / "manifest.json"
    candidate = freeze_dir.parent / ".h1_benchmark_v1.candidate.jsonl"
    write_jsonl(candidate, records)
    candidate_hash = sha256_file(candidate)
    if benchmark.exists():
        if sha256_file(benchmark) != candidate_hash:
            raise FileExistsError("immutable h1_benchmark_v1 already exists with different content")
    else:
        benchmark.write_bytes(candidate.read_bytes())
    candidate.unlink()
    phases = {phase: sum(phase_of(record) == phase for record in records) for phase in ("mid", "late")}
    proof_phases = {phase: sum(phase_of(record) == phase and bool(record["valid_proofs"]) for record in records) for phase in ("mid", "late")}
    manifest = {
        "benchmark_version": "h1_benchmark_v1",
        "benchmark_file": benchmark.name,
        "benchmark_sha256": candidate_hash,
        "n": len(records),
        "phase_counts": phases,
        "proof_bearing_phase_counts": proof_phases,
        "content": "94 exact_complete 6x6/k=4 gold states; partial and large-board states excluded",
        "schema_versions": sorted({record["schema_version"] for record in records}),
        "generator_versions": sorted({record["provenance"]["generator_version"] for record in records}),
        "seeds": sorted({int(record["provenance"]["seed"]) for record in records}),
        "budget_star_ms": {"6x6": 2000},
        "dedup_mode": "d4",
        "proof_annotation_version": ANNOTATION_VERSION,
        "proof_replay": annotation_stats,
        "exact_agreement": exact_gate,
        "frozen_date": "2026-08-14",
        "immutable_rule": "never overwrite; changes require h1_benchmark_v2",
        "source_artifact": str(source),
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise FileExistsError("immutable manifest already exists with different content")
    else:
        write_json(manifest_path, manifest)
    os.chmod(benchmark, 0o444)
    os.chmod(manifest_path, 0o444)
    return benchmark, manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-dir", type=Path, default=Path("diagnostic/h1_benchmark_v1"))
    parser.add_argument("--rgat-checkpoint", type=Path, required=True)
    parser.add_argument("--rgcn-checkpoint", type=Path, required=True)
    parser.add_argument("--mcts-playouts", type=int, default=50)
    parser.add_argument("--vcf-budget-ms", type=int, default=2000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Step 4: evaluator contract and proof-aware denominators.
    source_records = load_gold_fail_closed(args.source)
    prepared, annotation_stats = annotate_gold(source_records, args.vcf_budget_ms)
    prepared_path = args.output_dir / "prepared_gold.jsonl"
    write_jsonl(prepared_path, prepared)
    evaluator_gate = evaluator_contract_gate(prepared)
    evaluator_gate["proof_annotation"] = annotation_stats
    write_json(args.output_dir / "evaluator_gate.json", evaluator_gate)

    # Step 5: D4/action gates must pass before any SVG is written.
    graph_gate = coordinate_gate(prepared)
    rgat_model, rgat_bundle = model_from_bundle(args.rgat_checkpoint, MODELS)
    if rgat_bundle["training_state"]["iteration"] != 60:
        raise RuntimeError("SVG checkpoint is not iter 60")
    representatives = {}
    runtime_checks = []
    for phase in ("mid", "late"):
        candidates = [record for record in prepared if phase_of(record) == phase and record["valid_proofs"]]
        if not candidates:
            raise RuntimeError(f"no proof-bearing {phase} state available for graph export")
        representatives[phase] = candidates[0]
        runtime_checks.append(runtime_mcts_action_gate(candidates[0], rgat_model))
    graph_gate["runtime_mcts_action_gate"] = runtime_checks
    graph_gate["representatives"] = {phase: record["state_id"] for phase, record in representatives.items()}
    write_json(args.output_dir / "graph_gate.json", graph_gate)
    if not graph_gate["passed"]:
        raise RuntimeError("graph gate failed")
    for phase, record in representatives.items():
        render_comparison_svg(record, rgat_model, args.output_dir / "figures" / f"{phase}_{record['state_id']}.svg")

    # Step 6: repeat original exact labels, then freeze immutable bytes + hash.
    exact_gate = exact_freeze_gate(prepared, args.output_dir / "freeze_exact_progress.json")
    write_json(args.output_dir / "freeze_exact_gate.json", exact_gate)
    benchmark, manifest_path, manifest = freeze_benchmark(
        prepared, args.source, args.freeze_dir, evaluator_gate, graph_gate, exact_gate, annotation_stats
    )

    # Step 7: endpoint-only evaluation on the already frozen benchmark.
    _, endpoint_summary = evaluate_models(
        benchmark,
        manifest_path,
        {"rgcn": args.rgcn_checkpoint, "rgat": args.rgat_checkpoint},
        args.output_dir,
        args.mcts_playouts,
    )
    final = {
        "benchmark": str(benchmark),
        "benchmark_sha256": manifest["benchmark_sha256"],
        "manifest": str(manifest_path),
        "evaluator_gate": evaluator_gate,
        "graph_gate": graph_gate,
        "exact_gate": exact_gate,
        "endpoint_summary": endpoint_summary,
    }
    write_json(args.output_dir / "e3b_summary.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
