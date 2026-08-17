"""Certified Gomoku 6x6/k=4 evaluation harness.

The harness evaluates frozen H1 records against one checkpoint or a whole H3 run.
It stays fail-closed on the certified exact-complete benchmark contract while
still accepting legacy schema-v1 tactical records after normalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from azgomoku.explanation.model_evidence import collect_model_evidence
from azgomoku.graph import structural_edges
from azgomoku.h1_schema import state_from_record, validate_record
from azgomoku.h3_checkpoint import model_from_bundle
from azgomoku.metrics.attention import collapse_metrics
from azgomoku.metrics.semantic_alignment import aggregate_proofs, baselines, entropy
from azgomoku.mcts import search
from models.rgcn import RGCN
from models.rgat import RGAT


MODEL_CLASSES = {"rgcn": RGCN, "rgat": RGAT}

PREFERRED_FIELD_ORDER = (
    "benchmark_sha256",
    "benchmark_path",
    "checkpoint_sha256",
    "checkpoint",
    "checkpoint_iteration",
    "iteration",
    "seed",
    "model_type",
    "alignment_role",
    "state_id",
    "phase",
    "label_kind",
    "proof_available",
    "proof_count",
    "has_immediate_win",
    "has_mandatory_block",
    "has_simple_fork",
    "legal_count",
    "optimal_count",
    "policy_top1_action",
    "policy_top1_correct",
    "policy_top3_correct",
    "policy_optimal_mass",
    "policy_entropy",
    "value_prediction",
    "solver_value",
    "value_error",
    "value_mse",
    "value_sign_correct",
    "mcts_playouts",
    "mcts_top1_action",
    "mcts_top1_correct",
    "mcts_top3_correct",
    "mcts_optimal_mass",
    "mcts_entropy",
    "search_gain",
    "graph_critical_mass",
    "graph_precision_at_k",
    "graph_recall_at_k",
    "graph_auprc",
    "graph_critical_edges",
    "random_critical_mass",
    "random_precision_at_k",
    "random_recall_at_k",
    "random_auprc",
    "structural_critical_mass",
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
    "optimizer_updates",
    "selfplay_games_seen",
    "replay_size",
    "positions_seen",
    "wall_seconds",
)

SUMMARY_FIELDS = (
    "policy_top1_correct",
    "policy_top3_correct",
    "policy_optimal_mass",
    "policy_entropy",
    "value_error",
    "value_mse",
    "value_sign_correct",
    "mcts_top1_correct",
    "mcts_top3_correct",
    "mcts_optimal_mass",
    "mcts_entropy",
    "search_gain",
    "graph_critical_mass",
    "graph_precision_at_k",
    "graph_recall_at_k",
    "graph_auprc",
    "graph_critical_edges",
    "random_critical_mass",
    "random_precision_at_k",
    "random_recall_at_k",
    "random_auprc",
    "structural_critical_mass",
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def write_rows(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return
    fieldnames = list(PREFERRED_FIELD_ORDER)
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    fieldnames.extend(extras)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def first_present(row: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def infer_ply(record: dict) -> int:
    provenance = record.get("provenance") or {}
    if provenance.get("ply") is not None:
        return int(provenance["ply"])
    state = record["state"]
    board_size = int(state["board_size"])
    legal_actions = state.get("legal_actions")
    if legal_actions is None:
        legal_actions = [
            index
            for index, value in enumerate(np.asarray(state["board"], dtype=np.int8).reshape(-1))
            if int(value) == 0
        ]
    return board_size * board_size - len(legal_actions)


def phase_of_record(record: dict) -> str:
    ply = infer_ply(record)
    if 5 <= ply <= 9:
        return "mid"
    if ply >= 10:
        return "late"
    raise ValueError(f"gold state outside certified phases: ply={ply}")


def load_certified_records(path: Path, *, limit: int | None = None) -> list[dict]:
    records: list[dict] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        validation = validate_record(raw)
        if not validation.accepted or not validation.eligible:
            raise ValueError(f"record {line_number} rejected: {validation.errors}")
        record = validation.record
        record.pop("_validation", None)
        solver = record.get("solver", {})
        if (
            solver.get("status") != "exact_complete"
            or solver.get("method") != "full_minimax"
            or not solver.get("optimal_actions_complete", False)
        ):
            raise ValueError(f"record {line_number} is not exact-complete gold")
        state = state_from_record(record)
        if state.size != 6 or state.win_length != 4:
            raise ValueError(f"record {line_number} is not 6x6/k=4")
        provenance = dict(record.get("provenance") or {})
        provenance.setdefault("ply", infer_ply(record))
        record["provenance"] = provenance
        phase_of_record(record)
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise ValueError("benchmark is empty")
    if len({record["state_id"] for record in records}) != len(records):
        raise ValueError("duplicate state_id in benchmark")
    return records


def topk_legal_actions(scores: np.ndarray, legal_actions: list[int], k: int) -> list[int]:
    if not legal_actions:
        return []
    k = max(1, min(int(k), len(legal_actions)))
    order = np.argsort(np.asarray(scores, dtype=float))
    selected = order[-k:][::-1]
    return [int(legal_actions[index]) for index in selected]


def sign_of(value: float, *, epsilon: float = 1e-6) -> int:
    if abs(value) <= epsilon:
        return 0
    return 1 if value > 0 else -1


def proof_concepts(record: dict) -> set[str]:
    concepts: set[str] = set()
    for proof in record.get("valid_proofs", []):
        for concept in proof.get("concepts", []):
            concepts.add(str(concept))
    return concepts


def evaluate_record(
    record: dict,
    model: torch.nn.Module,
    model_type: str,
    checkpoint: Path,
    checkpoint_hash: str,
    benchmark_hash: str,
    benchmark_path: Path,
    playouts: int,
) -> dict:
    state = state_from_record(record)
    legal_actions = list(map(int, state.legal_actions()))
    optimal_actions = tuple(map(int, record["solver"]["optimal_actions"]))
    if not legal_actions:
        raise ValueError(f"terminal state has no legal actions: {record['state_id']}")
    if not optimal_actions:
        raise ValueError(f"record has no optimal actions: {record['state_id']}")
    optimal_set = set(optimal_actions)
    device = next(model.parameters()).device
    x = torch.from_numpy(state.features()).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = model(x, return_evidence=False)

    masked_logits = torch.full_like(logits, -torch.inf)
    masked_logits[0, legal_actions] = logits[0, legal_actions]
    policy = torch.softmax(masked_logits, dim=-1)[0].cpu().numpy()
    policy_top3 = topk_legal_actions(policy, legal_actions, 3)
    policy_action = policy_top3[0] if policy_top3 else int(legal_actions[0])

    selected_action = policy_action if policy_action in optimal_set else min(optimal_set)
    evidence = collect_model_evidence(state, model, selected_action)

    if model_type == "rgat":
        edges = evidence["graph_evidence"]["edges"]
        scores = [float(edge["attention"]) for edge in edges]
        collapse = collapse_metrics(edges)
    else:
        edges = structural_edges(state.size)
        scores = [float(edge["attention"]) for edge in edges]
        collapse = {
            "attention_normalized_entropy": None,
            "attention_structural_mae": None,
            "attention_head_diversity": None,
            "attention_topology_correlation": None,
            "attention_collapse_flag": None,
        }

    proofs = record.get("valid_proofs", [])
    if proofs:
        alignment = aggregate_proofs(edges, scores, proofs)["mean"]
        random_base, structural_base = baselines(edges, proofs, record["state_id"])
    else:
        alignment = {
            "mass": None,
            "precision_at_k": None,
            "recall_at_k": None,
            "auprc": None,
            "critical_edges": None,
        }
        random_base = {
            "mass": None,
            "precision_at_k": None,
            "recall_at_k": None,
            "auprc": None,
        }
        structural_base = {
            "mass": None,
            "precision_at_k": None,
            "recall_at_k": None,
            "auprc": None,
        }

    value_prediction = float(value.item())
    solver_value = int(record["solver"]["value"])
    value_error = abs(value_prediction - solver_value)

    row = {
        "benchmark_sha256": benchmark_hash,
        "benchmark_path": str(benchmark_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint": str(checkpoint),
        "checkpoint_iteration": int(record.get("_checkpoint_iteration", -1)),
        "iteration": int(record.get("_checkpoint_iteration", -1)),
        "seed": record.get("_checkpoint_seed"),
        "model_type": model_type,
        "alignment_role": "learned_attention_subject" if model_type == "rgat" else "structural_baseline_by_design_not_finding",
        "state_id": record["state_id"],
        "phase": phase_of_record(record),
        "label_kind": record.get("_validation", {}).get("label_kind") or record["solver"]["status"],
        "proof_available": int(bool(proofs)),
        "proof_count": len(proofs),
        "has_immediate_win": int("immediate_win" in proof_concepts(record)),
        "has_mandatory_block": int("mandatory_block" in proof_concepts(record)),
        "has_simple_fork": int("simple_fork" in proof_concepts(record)),
        "legal_count": len(legal_actions),
        "optimal_count": len(optimal_actions),
        "policy_top1_action": int(policy_action),
        "policy_top1_correct": int(policy_action in optimal_set),
        "policy_top3_correct": int(any(action in optimal_set for action in policy_top3)),
        "policy_optimal_mass": float(sum(policy[action] for action in optimal_actions)),
        "policy_entropy": entropy(policy[legal_actions]),
        "value_prediction": value_prediction,
        "solver_value": solver_value,
        "value_error": value_error,
        "value_mse": value_error * value_error,
        "value_sign_correct": int(sign_of(value_prediction) == sign_of(float(solver_value))),
        "mcts_playouts": int(playouts) if playouts > 0 else 0,
        "mcts_top1_action": None,
        "mcts_top1_correct": None,
        "mcts_top3_correct": None,
        "mcts_optimal_mass": None,
        "mcts_entropy": None,
        "search_gain": None,
        "graph_critical_mass": alignment["mass"],
        "graph_precision_at_k": alignment["precision_at_k"],
        "graph_recall_at_k": alignment["recall_at_k"],
        "graph_auprc": alignment["auprc"],
        "graph_critical_edges": alignment["critical_edges"],
        "random_critical_mass": random_base["mass"],
        "random_precision_at_k": random_base["precision_at_k"],
        "random_recall_at_k": random_base["recall_at_k"],
        "random_auprc": random_base["auprc"],
        "structural_critical_mass": structural_base["mass"],
        "structural_precision_at_k": structural_base["precision_at_k"],
        "structural_recall_at_k": structural_base["recall_at_k"],
        "structural_auprc": structural_base["auprc"],
        "alignment_minus_random": None,
        "alignment_minus_structural": None,
        "attention_normalized_entropy": collapse["attention_normalized_entropy"],
        "attention_structural_mae": collapse["attention_structural_mae"],
        "attention_head_diversity": collapse["attention_head_diversity"],
        "attention_topology_correlation": collapse["attention_topology_correlation"],
        "attention_collapse_flag": collapse["attention_collapse_flag"],
    }

    if alignment["mass"] is not None and random_base["mass"] is not None:
        row["alignment_minus_random"] = float(alignment["mass"]) - float(random_base["mass"])
    if alignment["mass"] is not None and structural_base["mass"] is not None:
        row["alignment_minus_structural"] = float(alignment["mass"]) - float(structural_base["mass"])

    if playouts > 0:
        pi, _ = search(model, state, playouts=playouts, temperature=1.0, return_root=True)
        mcts_top3 = topk_legal_actions(pi, legal_actions, 3)
        mcts_action = mcts_top3[0] if mcts_top3 else int(legal_actions[0])
        row.update(
            {
                "mcts_top1_action": int(mcts_action),
                "mcts_top1_correct": int(mcts_action in optimal_set),
                "mcts_top3_correct": int(any(action in optimal_set for action in mcts_top3)),
                "mcts_optimal_mass": float(sum(pi[action] for action in optimal_actions)),
                "mcts_entropy": entropy(pi[legal_actions]),
                "search_gain": float(sum(pi[action] for action in optimal_actions) - sum(policy[action] for action in optimal_actions)),
            }
        )

    return row


def load_model_bundle(path: Path) -> tuple[torch.nn.Module, dict, int, str, str]:
    model, bundle = model_from_bundle(path, MODEL_CLASSES)
    checkpoint_iteration = int(bundle["training_state"]["iteration"])
    checkpoint_sha256 = sha256_file(path)
    return model, bundle, checkpoint_iteration, bundle["model_type"], checkpoint_sha256


def checkpoint_paths_from_run_dir(run_dir: Path) -> list[Path]:
    manifest_path = run_dir / "checkpoints" / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint_dir = run_dir / "checkpoints"
        paths = [checkpoint_dir / item["path"] for item in manifest.get("checkpoints", [])]
        if paths:
            return paths
    checkpoint_dir = run_dir / "checkpoints"
    paths = sorted(checkpoint_dir.glob("iter_*.pt"))
    if paths:
        return paths
    raise FileNotFoundError(f"no checkpoint bundles found under {run_dir}")


def evaluate_checkpoint(
    checkpoint: Path,
    records: list[dict],
    benchmark_hash: str,
    benchmark_path: Path,
    playouts: int,
) -> tuple[list[dict], dict]:
    start = time.perf_counter()
    model, bundle, checkpoint_iteration, model_type, checkpoint_hash = load_model_bundle(checkpoint)
    rows = []
    for record in records:
        row = evaluate_record(
            record,
            model,
            model_type,
            checkpoint,
            checkpoint_hash,
            benchmark_hash,
            benchmark_path,
            playouts,
        )
        row["checkpoint_iteration"] = checkpoint_iteration
        row["iteration"] = checkpoint_iteration
        row["seed"] = bundle.get("seed")
        row["optimizer_updates"] = bundle.get("training_state", {}).get("optimizer_updates")
        row["selfplay_games_seen"] = bundle.get("training_state", {}).get("selfplay_games_seen")
        row["replay_size"] = bundle.get("training_state", {}).get("replay_size")
        row["positions_seen"] = bundle.get("training_state", {}).get("positions_seen")
        rows.append(row)
    elapsed = time.perf_counter() - start
    checkpoint_summary = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "iteration": checkpoint_iteration,
        "model_type": model_type,
        "rows": len(rows),
        "wall_seconds": elapsed,
    }
    return rows, checkpoint_summary


def mean_or_none(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def group_label(rows: list[dict]) -> str:
    model_type = rows[0]["model_type"]
    phase = rows[0]["phase"]
    iterations = sorted({int(row["iteration"]) for row in rows if row.get("iteration") is not None})
    if len(iterations) > 1:
        return f"{model_type}:iter{iterations[0]}-{iterations[-1]}:{phase}"
    iteration = iterations[0] if iterations else "na"
    return f"{model_type}:iter{iteration}:{phase}"


def summarize(rows: list[dict], benchmark_hash: str, benchmark_path: Path, checkpoints: list[dict]) -> dict:
    overall = {
        "n": len(rows),
        "proof_bearing": sum(int(row["proof_available"]) for row in rows),
        "concept_counts": {
            "immediate_win": sum(int(row["has_immediate_win"]) for row in rows),
            "mandatory_block": sum(int(row["has_mandatory_block"]) for row in rows),
            "simple_fork": sum(int(row["has_simple_fork"]) for row in rows),
        },
        "means": {field: mean_or_none(rows, field) for field in SUMMARY_FIELDS},
    }
    summary = {
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": benchmark_hash,
        "rows": len(rows),
        "checkpoints": checkpoints,
        "overall": overall,
        "groups": {},
        "concepts": {},
        "scope_notes": [
            "Certified exact-complete 6x6/k=4 states only.",
            "Legacy schema-v1 tactical records are normalized only after validation.",
            "R-GCN graph alignment is structural by design; it is not a learned-attention finding.",
        ],
    }

    multi_iteration = len({int(row["iteration"]) for row in rows if row.get("iteration") is not None}) > 1
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        parts = [row["model_type"]]
        if multi_iteration:
            parts.append(f"iter{int(row['iteration'])}")
        parts.append(row["phase"])
        grouped[":".join(parts)].append(row)

    for key, group in grouped.items():
        summary["groups"][key] = {
            "n": len(group),
            "proof_bearing": sum(int(row["proof_available"]) for row in group),
            "concept_counts": {
                "immediate_win": sum(int(row["has_immediate_win"]) for row in group),
                "mandatory_block": sum(int(row["has_mandatory_block"]) for row in group),
                "simple_fork": sum(int(row["has_simple_fork"]) for row in group),
            },
            "means": {field: mean_or_none(group, field) for field in SUMMARY_FIELDS},
        }

    for concept in ("immediate_win", "mandatory_block", "simple_fork"):
        subset = [row for row in rows if row.get(f"has_{concept}")]
        summary["concepts"][concept] = {
            "n": len(subset),
            "policy_top1_rate": mean_or_none(subset, "policy_top1_correct"),
            "policy_top3_rate": mean_or_none(subset, "policy_top3_correct"),
            "policy_optimal_mass": mean_or_none(subset, "policy_optimal_mass"),
            "mcts_top1_rate": mean_or_none(subset, "mcts_top1_correct"),
            "mcts_top3_rate": mean_or_none(subset, "mcts_top3_correct"),
            "mcts_optimal_mass": mean_or_none(subset, "mcts_optimal_mass"),
            "value_error": mean_or_none(subset, "value_error"),
            "value_mse": mean_or_none(subset, "value_mse"),
        }

    summary["vcf_find_rate"] = summary["concepts"]["immediate_win"]["policy_top1_rate"]
    summary["vcf_block_rate"] = summary["concepts"]["mandatory_block"]["policy_top1_rate"]
    summary["value_sign_accuracy"] = mean_or_none(rows, "value_sign_correct")
    return summary


def default_output_path(source: Path, *, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir / "eval_metrics.csv"
    return source.with_name(source.stem + ".eval.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="single checkpoint bundle .pt")
    source.add_argument("--run-dir", type=Path, help="run directory with checkpoints/manifest.json")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("diagnostic/h1_benchmark_v1/h1_benchmark_v1.jsonl"),
        help="certified H1 benchmark JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV or JSONL output path; default depends on mode",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="summary JSON path; defaults next to --output",
    )
    parser.add_argument("--playouts", type=int, default=50, help="MCTS playouts per record")
    parser.add_argument("--limit", type=int, default=None, help="optional benchmark record cap for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark)
    benchmark_hash = sha256_file(benchmark_path)
    records = load_certified_records(benchmark_path, limit=args.limit)

    if args.run_dir is not None:
        checkpoints = checkpoint_paths_from_run_dir(Path(args.run_dir))
        output_path = args.output or default_output_path(Path(args.run_dir), run_dir=Path(args.run_dir))
    else:
        checkpoints = [Path(args.checkpoint)]
        output_path = args.output or default_output_path(Path(args.checkpoint), run_dir=None)

    all_rows: list[dict] = []
    checkpoint_stats: list[dict] = []
    for checkpoint in checkpoints:
        rows, checkpoint_summary = evaluate_checkpoint(checkpoint, records, benchmark_hash, benchmark_path, args.playouts)
        all_rows.extend(rows)
        checkpoint_stats.append(checkpoint_summary)

    all_rows.sort(
        key=lambda row: (
            row["model_type"],
            int(row["iteration"]) if row.get("iteration") is not None else -1,
            row["phase"],
            row["state_id"],
        )
    )

    write_rows(output_path, all_rows)
    summary_path = args.summary or output_path.with_name(output_path.stem + ".summary.json")
    summary = summarize(all_rows, benchmark_hash, benchmark_path, checkpoint_stats)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
