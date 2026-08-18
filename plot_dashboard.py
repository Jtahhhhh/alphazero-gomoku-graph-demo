"""AlphaZero-style training dashboard for Gomoku runs.

The dashboard understands the repo's current CSV logs and also accepts JSONL
evaluation traces when the eval harness writes one. It groups repeated rows by
iteration and builds a dashboard with:
1) self-play health
2) training losses
3) evaluation quality
4) a dedicated game-length panel
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - friendly runtime error only
    plt = None
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


TRAIN_PANEL_TOP = ("selfplay_games_seen", "positions_seen", "replay_size")
TRAIN_PANEL_BOTTOM = ("selfplay_seconds", "training_seconds", "iteration_seconds")
GAME_LENGTH_PANEL = ("game_length",)
LOSS_PANEL_TOP = ("policy_loss", "value_loss", "total_loss")
LOSS_PANEL_BOTTOM = ("policy_entropy", "opening_entropy", "opening_corner_mass", "opening_edge_mass")
EVAL_PANEL_TOP = ("policy_top1_correct", "policy_top3_correct", "policy_optimal_mass", "mcts_top1_correct", "mcts_top3_correct", "mcts_optimal_mass")
EVAL_PANEL_BOTTOM = (
    "value_error",
    "value_mse",
    "search_gain",
    "graph_critical_mass",
    "graph_auprc",
    "attention_topology_correlation",
    "attention_collapse_flag",
)


def coerce_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered == "true":
        return 1
    if lowered == "false":
        return 0
    if re.fullmatch(r"[-+]?\d+", text):
        try:
            return int(text)
        except ValueError:
            return text
    try:
        return float(text)
    except ValueError:
        return text


def read_table(path: Path) -> list[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".tsv":
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return list(data)
        if isinstance(data, dict):
            for key in ("rows", "data", "records"):
                if isinstance(data.get(key), list):
                    return list(data[key])
        raise ValueError(f"unsupported JSON structure in {path}")
    raise ValueError(f"unsupported log format: {path}")


def normalize_rows(rows: list[dict]) -> list[dict]:
    normalized = []
    for row in rows:
        normalized.append({key: coerce_value(value) for key, value in row.items()})
    return normalized


def guess_iteration(row: dict, fallback_index: int) -> int:
    for key in ("iteration", "training_step", "step", "checkpoint_iteration", "update_in_iteration"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    checkpoint = row.get("checkpoint") or row.get("checkpoint_path")
    if isinstance(checkpoint, str):
        match = re.search(r"iter_(\d+)", checkpoint)
        if match:
            return int(match.group(1))
    return fallback_index


def source_label(path: Path) -> str:
    path = Path(path)
    parts = path.parts
    for marker in ("results", "diagnostic"):
        if marker in parts:
            index = parts.index(marker)
            relative = parts[index + 1 :]
            if len(relative) > 1:
                return "/".join(relative[:-1])
            if relative:
                return relative[0]
    return path.stem


def split_labels(path: Path, rows: list[dict]) -> dict[str, list[dict]]:
    label = source_label(path)
    split_fields = [
        field
        for field in ("model_type", "phase")
        if len({row.get(field) for row in rows if row.get(field) not in (None, "")}) > 1
    ]
    if not split_fields:
        return {label: rows}
    grouped: dict[tuple[Any, ...], list[dict]] = defaultdict(list)
    for row in rows:
        key = tuple(row.get(field) for field in split_fields)
        grouped[key].append(row)
    output = {}
    for key, group in grouped.items():
        suffix = "/".join(str(value) for value in key if value not in (None, ""))
        output[f"{label} | {suffix}"] = group
    return output


def aggregate_rows(rows: list[dict]) -> list[dict]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        buckets[guess_iteration(row, index)].append(row)
    aggregated = []
    for iteration, bucket in sorted(buckets.items()):
        result = {"iteration": iteration}
        keys = {key for row in bucket for key in row.keys()}
        for key in keys:
            values = [row.get(key) for row in bucket if row.get(key) not in (None, "")]
            if not values:
                continue
            numeric = []
            all_numeric = True
            for value in values:
                if isinstance(value, (int, float, bool)):
                    numeric.append(float(value))
                else:
                    all_numeric = False
                    break
            if all_numeric:
                result[key] = float(np.mean(numeric))
            else:
                result[key] = values[0]
        aggregated.append(result)
    return aggregated


def load_series(paths: list[Path] | None) -> dict[str, list[dict]]:
    if paths:
        selected = [Path(path) for path in paths]
    else:
        selected = sorted(Path("results").glob("**/training_log.csv"))
        if not selected:
            raise FileNotFoundError("no training logs found; pass --train explicitly")
    series: dict[str, list[dict]] = {}
    for path in selected:
        rows = aggregate_rows(normalize_rows(read_table(path)))
        for label, group in split_labels(path, rows).items():
            unique_label = label
            if unique_label in series:
                suffix = path.stem
                candidate = f"{label} | {suffix}"
                counter = 2
                while candidate in series:
                    candidate = f"{label} | {suffix} #{counter}"
                    counter += 1
                unique_label = candidate
            series[unique_label] = group
    return series


def metric_color(metric: str) -> str:
    palette = {
        "selfplay_games_seen": "#0f766e",
        "positions_seen": "#2563eb",
        "replay_size": "#7c3aed",
        "selfplay_seconds": "#dc2626",
        "training_seconds": "#f97316",
        "iteration_seconds": "#0ea5e9",
        "policy_loss": "#2563eb",
        "value_loss": "#dc2626",
        "total_loss": "#0f766e",
        "policy_entropy": "#7c3aed",
        "opening_entropy": "#8b5cf6",
        "opening_corner_mass": "#ea580c",
        "opening_edge_mass": "#16a34a",
        "policy_top1_correct": "#2563eb",
        "policy_top3_correct": "#7c3aed",
        "policy_optimal_mass": "#0f766e",
        "mcts_top1_correct": "#dc2626",
        "mcts_top3_correct": "#ea580c",
        "mcts_optimal_mass": "#16a34a",
        "value_error": "#dc2626",
        "value_mse": "#ea580c",
        "search_gain": "#0f766e",
        "graph_critical_mass": "#2563eb",
        "graph_auprc": "#7c3aed",
        "attention_topology_correlation": "#0f766e",
        "attention_collapse_flag": "#dc2626",
        "game_length": "#16a34a",
    }
    return palette.get(metric, "#334155")


def metric_style(index: int) -> str:
    return ("-" if index == 0 else "--" if index == 1 else ":" if index == 2 else "-.") if index < 4 else "-"


def available_metrics(rows: list[dict], candidates: tuple[str, ...]) -> list[str]:
    present = []
    keys = {key for row in rows for key in row.keys()}
    for candidate in candidates:
        if candidate in keys:
            present.append(candidate)
    return present


def plot_panel(ax, series: dict[str, list[dict]], metrics: tuple[str, ...], title: str, ylabel: str) -> None:
    plotted = False
    for series_index, (label, rows) in enumerate(series.items()):
        x = [row["iteration"] for row in rows]
        for metric in available_metrics(rows, metrics):
            y = [row.get(metric) for row in rows]
            points = [(float(xv), float(yv)) for xv, yv in zip(x, y) if yv is not None]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(
                xs,
                ys,
                color=metric_color(metric),
                linestyle=metric_style(series_index),
                linewidth=2.0,
                alpha=0.9,
                label=f"{metric} | {label}" if len(series) > 1 else metric,
            )
            plotted = True
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Iteration")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.set_axisbelow(True)
    if plotted:
        ax.legend(fontsize=8, frameon=False, loc="best")
    else:
        ax.text(0.5, 0.5, "No matching metrics", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#64748b")


def build_dashboard(train_series: dict[str, list[dict]], eval_series: dict[str, list[dict]], output: Path, title: str, dpi: int) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib is required for plot_dashboard.py: {_MATPLOTLIB_ERROR}")

    fig = plt.figure(figsize=(18, 11), dpi=dpi, facecolor="#f8fafc")
    grid = fig.add_gridspec(2, 3, hspace=0.28, wspace=0.18)

    left_stack = grid[1, 0].subgridspec(2, 1, hspace=0.32)
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(left_stack[0, 0]),
        fig.add_subplot(left_stack[1, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, 1]),
        fig.add_subplot(grid[0, 2]),
        fig.add_subplot(grid[1, 2]),
    ]

    plot_panel(axes[0], train_series, TRAIN_PANEL_TOP, "Self-play progress", "Count")
    plot_panel(axes[1], train_series, TRAIN_PANEL_BOTTOM, "Self-play timing", "Seconds")
    plot_panel(axes[2], train_series, GAME_LENGTH_PANEL, "Game length", "Moves")
    plot_panel(axes[3], train_series, LOSS_PANEL_TOP, "Training loss", "Loss")
    plot_panel(axes[4], train_series, LOSS_PANEL_BOTTOM, "Training entropy / openings", "Value")
    plot_panel(axes[5], eval_series or train_series, EVAL_PANEL_TOP, "Evaluation quality", "Score")
    plot_panel(axes[6], eval_series or train_series, EVAL_PANEL_BOTTOM, "Evaluation diagnostics", "Value")

    fig.suptitle(title, fontsize=16, fontweight="bold", color="#0f172a")
    fig.text(0.5, 0.02, "Iteration", ha="center", fontsize=11, color="#334155")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", facecolor=fig.get_facecolor())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        action="append",
        default=None,
        help="training log CSV/JSONL (repeatable). If omitted, results/**/training_log.csv is used.",
    )
    parser.add_argument(
        "--eval",
        type=Path,
        action="append",
        default=None,
        help="evaluation log CSV/JSONL/JSON (repeatable).",
    )
    parser.add_argument("--output", type=Path, default=Path("dashboard.png"), help="output image path")
    parser.add_argument("--title", type=str, default="AlphaZero-style training dashboard")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_series = load_series(args.train)
    eval_series = load_series(args.eval) if args.eval else {}
    build_dashboard(train_series, eval_series, Path(args.output), args.title, args.dpi)
    print(json.dumps({"output": str(Path(args.output)), "train_series": list(train_series), "eval_series": list(eval_series)}, indent=2))


if __name__ == "__main__":
    main()
