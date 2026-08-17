"""Dashboard plotting for Gomoku arena training and evaluation."""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_training_logs(log_dir: Path) -> Dict[str, list]:
    """
    Load training logs from CSV files in checkpoint directories.

    Args:
        log_dir: Base results directory containing model checkpoints

    Returns:
        Dict mapping model_name to list of training records
    """
    training_data = {}

    # Check for training_log.csv in subdirectories
    for model_dir in log_dir.glob("*/"):
        if not model_dir.is_dir():
            continue

        log_file = model_dir / "training_log.csv"
        if log_file.exists():
            records = []
            with open(log_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)

            model_name = model_dir.name
            training_data[model_name] = records

    return training_data


def load_eval_logs(log_dir: Path) -> Dict[str, List[dict]]:
    """
    Load evaluation logs from JSON files.

    Args:
        log_dir: Directory containing eval logs

    Returns:
        Dict mapping model_name to list of eval records
    """
    eval_data = {}

    if not log_dir.exists():
        return eval_data

    for json_file in log_dir.glob("*_summary.json"):
        # Extract model name from filename
        parts = json_file.stem.split("_iter")
        if len(parts) >= 2:
            model_name = parts[0]
            if model_name not in eval_data:
                eval_data[model_name] = []

            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                eval_data[model_name].append(data)

    # Sort eval data by iteration
    for model_name in eval_data:
        eval_data[model_name].sort(key=lambda x: x.get("iteration", 0))

    return eval_data


def plot_training_loss(training_data: Dict[str, list], output_dir: Path) -> None:
    """
    Plot training loss over iterations.

    Args:
        training_data: Dict mapping model_name to training records
        output_dir: Directory to save figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    for model_name, records in training_data.items():
        if not records:
            continue

        # Extract policy loss by iteration
        iterations = {}
        for record in records:
            it = int(record.get("training_iteration", 0))
            pl = float(record.get("policy_loss", 0))
            if it not in iterations:
                iterations[it] = {"policy": [], "value": []}
            iterations[it]["policy"].append(pl)
            iterations[it]["value"].append(float(record.get("value_loss", 0)))

        # Average across epochs within iteration
        its = sorted(iterations.keys())
        policy_loss = [np.mean(iterations[it]["policy"]) for it in its]
        value_loss = [np.mean(iterations[it]["value"]) for it in its]

        ax1.plot(its, policy_loss, marker="o", label=model_name, alpha=0.7)
        ax2.plot(its, value_loss, marker="s", label=model_name, alpha=0.7)

    ax1.set_xlabel("Training Iteration")
    ax1.set_ylabel("Policy Loss")
    ax1.set_title("Policy Loss Over Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Training Iteration")
    ax2.set_ylabel("Value Loss")
    ax2.set_title("Value Loss Over Training")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "training_loss.png", dpi=150)
    plt.close()

    print(f"Saved training_loss.png to {output_dir}")


def plot_eval_winrate(eval_data: Dict[str, List[dict]], output_dir: Path) -> None:
    """
    Plot win rates vs opponents over iterations.

    Args:
        eval_data: Dict mapping model_name to eval records
        output_dir: Directory to save figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by opponent type and strength
    opponent_configs = {}

    for model_name, records in eval_data.items():
        for record in records:
            for opp_key, opp_results in record.get("results_by_opponent", {}).items():
                if opp_key not in opponent_configs:
                    opponent_configs[opp_key] = {}
                if model_name not in opponent_configs[opp_key]:
                    opponent_configs[opp_key][model_name] = []

                opponent_configs[opp_key][model_name].append(
                    {
                        "iteration": record.get("iteration", 0),
                        "win_rate": opp_results.get("win_rate", 0),
                    }
                )

    # Plot each opponent type
    for opp_key, model_data in opponent_configs.items():
        fig, ax = plt.subplots(figsize=(12, 6))

        for model_name, data_points in model_data.items():
            data_points.sort(key=lambda x: x["iteration"])
            iterations = [d["iteration"] for d in data_points]
            win_rates = [d["win_rate"] for d in data_points]

            ax.plot(iterations, win_rates, marker="o", label=model_name, alpha=0.7)

        # Add threshold line
        ax.axhline(y=0.95, color="r", linestyle="--", label="95% threshold", alpha=0.5)

        ax.set_xlabel("Training Iteration")
        ax.set_ylabel("Win Rate")
        ax.set_title(f"Win Rate vs {opp_key}")
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"winrate_vs_{opp_key}.png", dpi=150)
        plt.close()

        print(f"Saved winrate_vs_{opp_key}.png to {output_dir}")


def plot_milestones(eval_data: Dict[str, List[dict]], output_dir: Path) -> None:
    """
    Identify and plot milestones (when models reach 95% win rate).

    Args:
        eval_data: Dict mapping model_name to eval records
        output_dir: Directory to save summary
    """
    from azgomoku.tracking.match_tracker import check_milestone

    milestones = {}

    for model_name, records in eval_data.items():
        milestones[model_name] = {}

        for record in records:
            for opp_key, opp_results in record.get("results_by_opponent", {}).items():
                if opp_key not in milestones[model_name]:
                    milestones[model_name][opp_key] = {
                        "win_rates": [],
                        "iterations": [],
                    }

                milestones[model_name][opp_key]["win_rates"].append(
                    opp_results.get("win_rate", 0)
                )
                milestones[model_name][opp_key]["iterations"].append(
                    record.get("iteration", 0)
                )

    # Check which milestones are reached
    summary = []
    for model_name in milestones:
        for opp_key in milestones[model_name]:
            win_rates = milestones[model_name][opp_key]["win_rates"]
            iterations = milestones[model_name][opp_key]["iterations"]

            if not win_rates:
                continue

            # Check 95% milestone with min 3 consecutive and 200 games (assuming ~40 games per eval)
            games_per_eval = 40
            milestone_idx = check_milestone(
                win_rates, games_per_eval, threshold=0.95, min_consecutive=3
            )

            if milestone_idx is not None:
                milestone_iter = iterations[milestone_idx]
                milestone_rate = win_rates[milestone_idx]
                summary.append(
                    {
                        "model": model_name,
                        "opponent": opp_key,
                        "milestone_iteration": milestone_iter,
                        "win_rate_at_milestone": milestone_rate,
                        "total_evals_to_milestone": milestone_idx + 3,
                    }
                )

    # Save milestone summary
    if summary:
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_file = output_dir / "milestones.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved milestones.json to {output_dir}")

        # Print summary
        print("\n=== Milestone Summary ===")
        for item in summary:
            print(
                f"{item['model']} vs {item['opponent']}: "
                f"Reached 95% at iteration {item['milestone_iteration']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Plot Gomoku arena training/eval dashboard")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Results directory containing model checkpoints",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("results/eval_logs"),
        help="Directory containing eval logs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures"),
        help="Output directory for plots",
    )
    args = parser.parse_args()

    print("Loading training logs...")
    training_data = load_training_logs(args.results_dir)

    print("Loading eval logs...")
    eval_data = load_eval_logs(args.eval_dir)

    if training_data:
        print("Plotting training loss...")
        plot_training_loss(training_data, args.output_dir)

    if eval_data:
        print("Plotting eval win rates...")
        plot_eval_winrate(eval_data, args.output_dir)
        print("Checking milestones...")
        plot_milestones(eval_data, args.output_dir)

    print("Dashboard generation complete!")


if __name__ == "__main__":
    main()
