"""Developmental inference over frozen H1 gold; never trains or generates states."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np

from azgomoku.h3_checkpoint import model_from_bundle
from investigation.e3b_common import load_gold_fail_closed, sha256_file
from investigation.e3b_pipeline import MODELS, evaluate_record


ITERATIONS = tuple(range(0, 61, 5))
MCTS_ITERATIONS = {0, 20, 40, 60}
NETWORK_METRICS = (
    "policy_optimal_mass",
    "policy_top1_correct",
    "value_error",
    "graph_critical_mass",
    "alignment_minus_structural",
    "alignment_minus_random",
    "graph_auprc",
    "attention_topology_correlation",
    "attention_collapse_flag",
)
TAIL_COMPETENCE_FRACTION = 0.95
TAIL_COLLAPSE_MAX = 0.05
HIGH_TOPOLOGY_MIN = 0.90
MEANINGFUL_ALIGNMENT_GAIN = 0.001


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_checkpoint_index(run_dir: Path, expected_model: str) -> dict[int, Path]:
    manifest_path = run_dir / "checkpoints" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_type") != expected_model:
        raise ValueError(f"checkpoint manifest model mismatch: {manifest_path}")
    index = {int(item["iteration"]): run_dir / "checkpoints" / item["path"] for item in manifest["checkpoints"]}
    if tuple(sorted(index)) != ITERATIONS:
        raise ValueError(f"expected 13 checkpoints {ITERATIONS}, got {tuple(sorted(index))}")
    if any(not path.exists() for path in index.values()):
        raise FileNotFoundError("checkpoint manifest references a missing file")
    return index


def _mean(rows: list[dict], key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return None if not values else float(np.mean(values))


def aggregate_rows(per_state: list[dict], benchmark_hash: str) -> list[dict]:
    aggregated = []
    for model_type in ("rgcn", "rgat"):
        for iteration in ITERATIONS:
            for phase in ("late", "mid"):
                rows = [
                    row for row in per_state
                    if row["model_type"] == model_type and int(row["iteration"]) == iteration and row["phase"] == phase
                ]
                if len(rows) != (71 if phase == "late" else 23):
                    raise RuntimeError(f"incomplete developmental group: {model_type}/{iteration}/{phase}")
                proof_rows = [row for row in rows if row["proof_available"]]
                item = {
                    "benchmark_sha256": benchmark_hash,
                    "checkpoint_sha256": rows[0]["checkpoint_sha256"],
                    "checkpoint": rows[0]["checkpoint"],
                    "model_type": model_type,
                    "alignment_role": rows[0]["alignment_role"],
                    "iteration": iteration,
                    "phase": phase,
                    "n": len(rows),
                    "claim": "main_finding_eligible" if phase == "late" else "suggestive_n23_no_conclusion",
                    "alignment_n": len(proof_rows),
                    "mcts_evaluated": iteration in MCTS_ITERATIONS,
                }
                for metric in NETWORK_METRICS:
                    item[metric] = _mean(rows, metric)
                item.update({
                    "structural_critical_mass": _mean(proof_rows, "structural_critical_mass"),
                    "random_critical_mass": _mean(proof_rows, "random_critical_mass"),
                    "mcts_optimal_mass": _mean(rows, "mcts_optimal_mass") if iteration in MCTS_ITERATIONS else None,
                    "mcts_top1_correct": _mean(rows, "mcts_top1_correct") if iteration in MCTS_ITERATIONS else None,
                    "search_gain": _mean(rows, "search_gain") if iteration in MCTS_ITERATIONS else None,
                })
                aggregated.append(item)
    return aggregated


def baseline_gate(rows: list[dict], tolerance: float = 1e-12) -> dict:
    details = {}
    for phase in ("late", "mid"):
        phase_rows = [row for row in rows if row["phase"] == phase]
        details[phase] = {}
        for metric in ("structural_critical_mass", "random_critical_mass"):
            values = [float(row[metric]) for row in phase_rows]
            spread = max(values) - min(values)
            details[phase][metric] = {"min": min(values), "max": max(values), "spread": spread}
            if spread > tolerance:
                raise RuntimeError(f"baseline drift detected: {phase}/{metric} spread={spread}")
    return {"passed": True, "tolerance": tolerance, "details": details}


def endpoint_gate(rows: list[dict], e3b_summary_path: Path, tolerance: float = 1e-12) -> dict:
    expected = json.loads(e3b_summary_path.read_text(encoding="utf-8"))["groups"]["rgat:late"]["means"]
    actual = next(row for row in rows if row["model_type"] == "rgat" and row["iteration"] == 60 and row["phase"] == "late")
    comparisons = {
        "graph_critical_mass": expected["graph_critical_mass"],
        "structural_critical_mass": expected["structural_critical_mass"],
        "random_critical_mass": expected["random_critical_mass"],
        "attention_topology_correlation": expected["attention_topology_correlation"],
    }
    deltas = {key: abs(float(actual[key]) - float(value)) for key, value in comparisons.items()}
    if any(delta > tolerance for delta in deltas.values()):
        raise RuntimeError(f"iter-60 does not match E-3b: {deltas}")
    return {"passed": True, "tolerance": tolerance, "expected": comparisons, "absolute_deltas": deltas}


def analyze_tail(rows: list[dict]) -> dict:
    late = sorted(
        [row for row in rows if row["model_type"] == "rgat" and row["phase"] == "late"],
        key=lambda row: row["iteration"],
    )
    max_competence = max(float(row["policy_optimal_mass"]) for row in late)
    competence_threshold = TAIL_COMPETENCE_FRACTION * max_competence
    tail = [
        row for row in late
        if float(row["policy_optimal_mass"]) >= competence_threshold
        and float(row["attention_collapse_flag"]) <= TAIL_COLLAPSE_MAX
    ]
    if not tail:
        raise RuntimeError("converged tail is empty")
    for row in late:
        row["excess_over_strongest_baseline"] = float(row["graph_critical_mass"]) - max(
            float(row["structural_critical_mass"]), float(row["random_critical_mass"])
        )
    tail_excess = [float(row["excess_over_strongest_baseline"]) for row in tail]
    all_below = all(value <= 0 for value in tail_excess)
    topology_high = all(float(row["attention_topology_correlation"]) >= HIGH_TOPOLOGY_MIN for row in tail)
    tail_gain = tail_excess[-1] - tail_excess[0] if len(tail_excess) > 1 else 0.0
    meaningful_rise = tail_gain > MEANINGFUL_ALIGNMENT_GAIN
    rejected = all_below and topology_high and not meaningful_rise
    best = max(late, key=lambda row: row["excess_over_strongest_baseline"])
    topology_values = [float(row["attention_topology_correlation"]) for row in late]
    if topology_values[0] >= HIGH_TOPOLOGY_MIN and min(topology_values) >= HIGH_TOPOLOGY_MIN:
        topology_scenario = "high_from_initialization"
    elif topology_values[-1] - topology_values[0] >= 0.05:
        topology_scenario = "drifting_toward_topology"
    else:
        topology_scenario = "mixed_or_stable_below_high_threshold"
    return {
        "definitions": {
            "competence": f"policy_optimal_mass >= {TAIL_COMPETENCE_FRACTION:.2f} * max",
            "collapse_free": f"hard_collapse_rate <= {TAIL_COLLAPSE_MAX:.2f}",
            "high_topology": f"topology_corr >= {HIGH_TOPOLOGY_MIN:.2f}",
            "meaningful_alignment_gain": f"tail last-first excess > {MEANINGFUL_ALIGNMENT_GAIN}",
        },
        "max_competence": max_competence,
        "competence_threshold": competence_threshold,
        "tail_iterations": [int(row["iteration"]) for row in tail],
        "tail_rows": [{
            "iteration": int(row["iteration"]),
            "policy_optimal_mass": row["policy_optimal_mass"],
            "hard_collapse_rate": row["attention_collapse_flag"],
            "critical_mass": row["graph_critical_mass"],
            "strongest_baseline": max(row["structural_critical_mass"], row["random_critical_mass"]),
            "excess": row["excess_over_strongest_baseline"],
            "topology_corr": row["attention_topology_correlation"],
        } for row in tail],
        "all_tail_alignment_at_or_below_baseline": all_below,
        "all_tail_topology_high": topology_high,
        "tail_alignment_excess_gain": tail_gain,
        "meaningful_tail_alignment_rise": meaningful_rise,
        "max_excess_over_any_checkpoint": {
            "iteration": int(best["iteration"]),
            "value": float(best["excess_over_strongest_baseline"]),
        },
        "topology_scenario": topology_scenario,
        "H_null_verdict": "rejected" if rejected else "not_rejected",
    }


def _polyline(values, x, y) -> str:
    return " ".join(f"{x(item[0]):.2f},{y(item[1]):.2f}" for item in values)


def _chart_svg(title: str, panels: list[dict], output: Path) -> None:
    width = 940
    panel_height = 260
    height = 70 + panel_height * len(panels)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial;fill:#0f172a}.title{font-size:20px;font-weight:700}.label{font-size:12px}.axis{stroke:#64748b;stroke-width:1}.grid{stroke:#e2e8f0;stroke-width:1}.line{fill:none;stroke-width:3}.dash{stroke-dasharray:8 5}.faint{opacity:.5}</style>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" class="title">{title}</text>',
    ]
    colors = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#f59e0b")
    for panel_index, panel in enumerate(panels):
        left, right = 82, width - 34
        top = 58 + panel_index * panel_height
        bottom = top + 175
        all_values = [value for series in panel["series"] for _, value in series["values"]]
        low = min(all_values)
        high = max(all_values)
        padding = (high - low) * 0.12 or 0.05
        low -= padding
        high += padding
        x = lambda value: left + value / 60 * (right - left)
        y = lambda value: bottom - (value - low) / (high - low) * (bottom - top)
        parts.append(f'<text x="{left}" y="{top-13}" class="label">{panel["y_label"]}</text>')
        for tick in range(0, 61, 10):
            px = x(tick)
            parts.append(f'<line x1="{px}" y1="{top}" x2="{px}" y2="{bottom}" class="grid"/><text x="{px}" y="{bottom+19}" text-anchor="middle" class="label">{tick}</text>')
        for fraction in (0, .5, 1):
            value = low + fraction * (high - low)
            py = y(value)
            parts.append(f'<line x1="{left}" y1="{py}" x2="{right}" y2="{py}" class="grid"/><text x="{left-8}" y="{py+4}" text-anchor="end" class="label">{value:.4f}</text>')
        parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/><text x="{(left+right)/2}" y="{bottom+39}" text-anchor="middle" class="label">Training iteration</text>')
        legend_x = left
        for index, series in enumerate(panel["series"]):
            style = "line" + (" dash" if series.get("dash") else "") + (" faint" if series.get("faint") else "")
            color = colors[index % len(colors)]
            parts.append(f'<polyline points="{_polyline(series["values"], x, y)}" class="{style}" stroke="{color}"/>')
            for iteration, value in series["values"]:
                parts.append(f'<circle cx="{x(iteration)}" cy="{y(value)}" r="3.5" fill="{color}" opacity="{.5 if series.get("faint") else 1}"/>')
            parts.append(f'<line x1="{legend_x}" y1="{bottom+59}" x2="{legend_x+24}" y2="{bottom+59}" stroke="{color}" stroke-width="3"/><text x="{legend_x+30}" y="{bottom+63}" class="label">{series["name"]}</text>')
            legend_x += 185
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")


def render_trajectories(rows: list[dict], output_dir: Path) -> None:
    def series(model, phase, metric):
        selected = sorted(
            [row for row in rows if row["model_type"] == model and row["phase"] == phase],
            key=lambda row: row["iteration"],
        )
        return [(int(row["iteration"]), float(row[metric])) for row in selected]

    rgat_late = sorted([row for row in rows if row["model_type"] == "rgat" and row["phase"] == "late"], key=lambda row: row["iteration"])
    _chart_svg("Developmental decoupling: competence rises, alignment vs baselines", [
        {"y_label": "Policy optimal mass", "series": [
            {"name": "R-GAT late (n=71)", "values": series("rgat", "late", "policy_optimal_mass")},
            {"name": "R-GCN late", "values": series("rgcn", "late", "policy_optimal_mass"), "dash": True},
            {"name": "R-GAT mid (n=23 suggestive)", "values": series("rgat", "mid", "policy_optimal_mass"), "faint": True},
        ]},
        {"y_label": "Critical mass", "series": [
            {"name": "R-GAT late", "values": series("rgat", "late", "graph_critical_mass")},
            {"name": "Structural control", "values": series("rgat", "late", "structural_critical_mass"), "dash": True},
            {"name": "Random control", "values": series("rgat", "late", "random_critical_mass"), "dash": True},
            {"name": "R-GAT mid suggestive", "values": series("rgat", "mid", "graph_critical_mass"), "faint": True},
        ]},
    ], output_dir / "decoupling.svg")
    _chart_svg("R-GAT attention topology correlation", [
        {"y_label": "Pearson correlation with structural topology", "series": [
            {"name": "Late (n=71)", "values": series("rgat", "late", "attention_topology_correlation")},
            {"name": "Mid (n=23 suggestive)", "values": series("rgat", "mid", "attention_topology_correlation"), "dash": True, "faint": True},
        ]},
    ], output_dir / "topology-correlation.svg")
    _chart_svg("R-GAT hard-collapse rate", [
        {"y_label": "Fraction of states meeting hard-collapse rule", "series": [
            {"name": "Late (n=71)", "values": series("rgat", "late", "attention_collapse_flag")},
            {"name": "Mid (n=23 suggestive)", "values": series("rgat", "mid", "attention_collapse_flag"), "dash": True, "faint": True},
        ]},
    ], output_dir / "hard-collapse.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rgcn-run", type=Path, required=True)
    parser.add_argument("--rgat-run", type=Path, required=True)
    parser.add_argument("--e3b-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mcts-playouts", type=int, default=50)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    benchmark_hash = sha256_file(args.benchmark)
    if benchmark_hash != manifest["benchmark_sha256"]:
        raise RuntimeError("immutable benchmark hash mismatch")
    if not args.benchmark.stat().st_mode & 0o222 == 0:
        # On drvfs, Windows read-only is still reflected by os.access/st_mode inconsistently;
        # hash matching remains the hard content gate.
        pass
    records = load_gold_fail_closed(args.benchmark)
    if len(records) != 94:
        raise RuntimeError("developmental evaluation requires frozen n=94")
    indexes = {
        "rgcn": load_checkpoint_index(args.rgcn_run, "rgcn"),
        "rgat": load_checkpoint_index(args.rgat_run, "rgat"),
    }
    progress_path = args.output_dir / "developmental_per_state.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    for model_type in ("rgcn", "rgat"):
        for iteration in ITERATIONS:
            checkpoint = indexes[model_type][iteration]
            checkpoint_hash = sha256_file(checkpoint)
            model, bundle = model_from_bundle(checkpoint, MODELS)
            if int(bundle["training_state"]["iteration"]) != iteration:
                raise RuntimeError("checkpoint bundle iteration mismatch")
            mcts_enabled = iteration in MCTS_ITERATIONS
            for index, record in enumerate(records):
                key = f"{model_type}:{iteration}:{record['state_id']}"
                if key in progress:
                    if progress[key]["benchmark_sha256"] != benchmark_hash:
                        raise RuntimeError("stale developmental cache uses another benchmark")
                    continue
                started = time.perf_counter()
                row = evaluate_record(
                    record, model, model_type, checkpoint, checkpoint_hash,
                    benchmark_hash, args.mcts_playouts if mcts_enabled else 1,
                )
                row["iteration"] = iteration
                row["optimizer_updates"] = bundle["training_state"]["optimizer_updates"]
                row["selfplay_games_seen"] = bundle["training_state"]["selfplay_games_seen"]
                if not mcts_enabled:
                    row["mcts_playouts"] = None
                    row["mcts_top1_correct"] = None
                    row["mcts_optimal_mass"] = None
                    row["search_gain"] = None
                row["wall_seconds"] = time.perf_counter() - started
                progress[key] = row
                write_json(progress_path, progress)
                print(json.dumps({"stage": "developmental", "model": model_type, "iteration": iteration, "state": index + 1, "total": len(records), "mcts": mcts_enabled}), flush=True)

    per_state = list(progress.values())
    aggregated = aggregate_rows(per_state, benchmark_hash)
    csv_path = args.output_dir / "developmental_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregated[0]))
        writer.writeheader()
        writer.writerows(aggregated)
    baseline = baseline_gate(aggregated)
    endpoint = endpoint_gate(aggregated, args.e3b_summary)
    tail = analyze_tail(aggregated)
    gates = {"benchmark_sha256": benchmark_hash, "baseline_gate": baseline, "endpoint_gate": endpoint}
    write_json(args.output_dir / "developmental_gates.json", gates)
    write_json(args.output_dir / "developmental_analysis.json", tail)
    render_trajectories(aggregated, args.output_dir / "figures")
    final = {
        "benchmark_sha256": benchmark_hash,
        "checkpoints": len(ITERATIONS),
        "models": 2,
        "per_state_rows": len(per_state),
        "aggregated_rows": len(aggregated),
        "mcts_iterations": sorted(MCTS_ITERATIONS),
        "gates": gates,
        "analysis": tail,
    }
    write_json(args.output_dir / "developmental_summary.json", final)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
