"""Reproducible Arena evaluation and playing-strength summaries."""

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from azgomoku.agents.alphabeta_agent import AlphaBetaAgent
from azgomoku.agents.egreedy_agent import EGreedyAgent
from azgomoku.game import GomokuState
from azgomoku.h3_checkpoint import model_from_bundle
from azgomoku.mcts import search
from azgomoku.reproducibility import seed_everything
from azgomoku.tensorboard_logging import create_writer
from models.cnn_baseline import CNNBaseline
from models.rgat import RGAT
from models.rgcn import RGCN


MODEL_CLASSES = {"cnn_baseline": CNNBaseline, "rgcn": RGCN, "rgat": RGAT}
OPPONENTS = ("egreedy", "alphabeta")
RAW_FIELDS = (
    "game_id", "model", "opponent", "seed", "model_side", "first_player",
    "result", "winner", "moves", "duration_seconds", "checkpoint_iteration",
    "mcts_playouts",
)
SUMMARY_FIELDS = (
    "model", "opponent", "games", "wins", "draws", "losses", "win_rate",
    "arena_score", "first_score", "second_score", "delta_elo",
)


def arena_score(wins, draws, losses):
    games = wins + draws + losses
    return (wins + 0.5 * draws) / games if games else 0.0


def win_rate(wins, draws, losses):
    games = wins + draws + losses
    return wins / games if games else 0.0


def delta_elo(score, games):
    if games <= 0:
        return 0.0
    eps = 0.5 / games
    bounded = max(eps, min(1.0 - eps, score))
    return 400.0 * math.log10(bounded / (1.0 - bounded))


def _opponent(kind, epsilon, depth, board_size, win_length, seed):
    if kind == "egreedy":
        return EGreedyAgent(epsilon=epsilon, board_size=board_size, win_length=win_length, seed=seed)
    if kind == "alphabeta":
        return AlphaBetaAgent(depth=depth, board_size=board_size, win_length=win_length)
    raise ValueError(f"unknown opponent: {kind}")


def play_game(model, opponent, board_size, win_length, mcts_playouts, model_first):
    state = GomokuState.initial(board_size, win_length)
    if not model_first:
        state = GomokuState(state.board.copy(), to_play=-1, last_move=-1, win_length=win_length)
    moves = []
    started = time.perf_counter()
    model_player = 1 if model_first else -1
    while not state.terminal():
        if state.to_play == model_player:
            action = int(search(model, state, mcts_playouts, temperature=0.0).argmax())
        else:
            action = opponent.select_move(state)
        moves.append(action)
        state = state.play(action)
    winner_value = state.winner()
    result = "win" if winner_value == model_player else "loss" if winner_value == -model_player else "draw"
    winner = "model" if winner_value == model_player else "opponent" if winner_value == -model_player else ""
    return result, winner, len(moves), time.perf_counter() - started


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fit_global_elo(matchups, anchor="egreedy"):
    agents = sorted({item["model"] for item in matchups} | {item["opponent"] for item in matchups})
    if anchor not in agents:
        raise ValueError(f"Elo anchor {anchor!r} is not present in Arena matchups")
    ratings = {agent: 0.0 for agent in agents}
    scale = math.log(10.0) / 400.0
    free = [agent for agent in agents if agent != anchor]
    for _ in range(100):
        max_change = 0.0
        for agent in free:
            gradient = 0.0
            information = 1e-9
            for item in matchups:
                if item["model"] != agent and item["opponent"] != agent:
                    continue
                model_rating = ratings[item["model"]]
                opponent_rating = ratings[item["opponent"]]
                expected = 1.0 / (1.0 + 10.0 ** ((opponent_rating - model_rating) / 400.0))
                actual = item["score"]
                direction = 1.0 if item["model"] == agent else -1.0
                gradient += direction * item["games"] * (actual - expected) * scale
                information += item["games"] * expected * (1.0 - expected) * scale * scale
            change = gradient / information
            ratings[agent] += change
            max_change = max(max_change, abs(change))
        if max_change < 1e-5:
            break
    return {agent: round(value, 3) for agent, value in ratings.items()}


def summarize(raw_rows):
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["model"], row["opponent"])].append(row)
    summaries = []
    matchups = []
    for (model, opponent), rows in sorted(grouped.items()):
        counts = {result: sum(row["result"] == result for row in rows) for result in ("win", "draw", "loss")}
        first = [row for row in rows if row["model_side"] == "black"]
        second = [row for row in rows if row["model_side"] == "white"]
        score = arena_score(counts["win"], counts["draw"], counts["loss"])
        item = {
            "model": model, "opponent": opponent, "games": len(rows),
            "wins": counts["win"], "draws": counts["draw"], "losses": counts["loss"],
            "win_rate": win_rate(counts["win"], counts["draw"], counts["loss"]),
            "arena_score": score,
            "first_score": arena_score(sum(r["result"] == "win" for r in first), sum(r["result"] == "draw" for r in first), sum(r["result"] == "loss" for r in first)),
            "second_score": arena_score(sum(r["result"] == "win" for r in second), sum(r["result"] == "draw" for r in second), sum(r["result"] == "loss" for r in second)),
        }
        item["delta_elo"] = delta_elo(score, len(rows))
        summaries.append(item)
        matchups.append({"model": model, "opponent": opponent, "games": len(rows), "score": score})
    return summaries, matchups


def run_arena(models, output, games=1000, board_size=15, win_length=5, mcts_playouts=400,
              epsilon=0.1, depth=2, seed=7, checkpoint_iteration=100, tensorboard_logdir=None):
    if games < 2 or games % 2:
        raise ValueError("games must be an even number so first/second sides are balanced")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    writer = create_writer(tensorboard_logdir) if tensorboard_logdir else None
    raw_rows = []
    for matchup_index, (model_name, model) in enumerate(models.items(), 1):
        model.eval()
        for opponent_name in OPPONENTS:
            print(f"[ARENA] {model_name} vs {opponent_name}", flush=True)
            counts = {"win": 0, "draw": 0, "loss": 0}
            for index in range(games):
                game_seed = seed + (matchup_index - 1) * games * len(OPPONENTS) + index
                seed_everything(game_seed)
                model_first = index < games // 2
                opponent = _opponent(opponent_name, epsilon, depth, board_size, win_length, game_seed)
                result, winner, moves, duration = play_game(model, opponent, board_size, win_length, mcts_playouts, model_first)
                counts[result] += 1
                raw_rows.append({
                    "game_id": f"{model_name}_vs_{opponent_name}_{index + 1:04d}",
                    "model": model_name, "opponent": opponent_name, "seed": game_seed,
                    "model_side": "black" if model_first else "white",
                    "first_player": model_name if model_first else opponent_name,
                    "result": result, "winner": model_name if winner == "model" else opponent_name if winner == "opponent" else "", "moves": moves,
                    "duration_seconds": round(duration, 6), "checkpoint_iteration": checkpoint_iteration,
                    "mcts_playouts": mcts_playouts,
                })
                if (index + 1) % max(1, games // 10) == 0:
                    score = arena_score(counts["win"], counts["draw"], counts["loss"])
                    print(f"[ARENA] game {index + 1}/{games} | W={counts['win']} D={counts['draw']} L={counts['loss']} | score={score:.1%}", flush=True)
            score = arena_score(counts["win"], counts["draw"], counts["loss"])
            print(f"[ARENA COMPLETE] model={model_name} opponent={opponent_name} games={games} W={counts['win']} D={counts['draw']} L={counts['loss']} arena_score={score:.1%} delta_elo={delta_elo(score, games):.1f}", flush=True)
    summaries, matchups = summarize(raw_rows)
    ratings = _fit_global_elo(matchups)
    reference = "egreedy"
    for item in summaries:
        item["win_rate"] = round(item["win_rate"], 6)
        item["arena_score"] = round(item["arena_score"], 6)
        item["first_score"] = round(item["first_score"], 6)
        item["second_score"] = round(item["second_score"], 6)
        item["delta_elo"] = round(item["delta_elo"], 3)
    _write_csv(output / "arena_games.csv", RAW_FIELDS, raw_rows)
    _write_csv(output / "arena_summary.csv", SUMMARY_FIELDS, summaries)
    (output / "arena_elo.json").write_text(json.dumps({"reference": {"agent": reference, "elo": ratings[reference]}, "ratings": ratings, "delta_vs_cnn": {name: round(value - ratings.get("cnn_baseline", 0.0), 3) for name, value in ratings.items()}}, indent=2), encoding="utf-8")
    if writer:
        for item in summaries:
            prefix=f"arena/{item['model']}_vs_{item['opponent']}"
            for metric in ("wins","draws","losses","win_rate","arena_score","first_score","second_score","delta_elo"):
                writer.add_scalar(f"{prefix}/{metric}",item[metric],checkpoint_iteration)
        for agent,rating in ratings.items():
            writer.add_scalar(f"elo/{agent}",rating,checkpoint_iteration)
            writer.add_scalar(f"evaluation/elo/{agent}",rating,checkpoint_iteration)
        for item in summaries:
            writer.add_scalar(f"evaluation/arena_score/{item['model']}_vs_{item['opponent']}",item["arena_score"],checkpoint_iteration)
        writer.flush(); writer.close()
    return summaries, ratings


def load_models(checkpoints, device="cpu"):
    loaded = {}
    for name, checkpoint in checkpoints.items():
        model, _ = model_from_bundle(checkpoint, MODEL_CLASSES)
        loaded[name] = model.to(device)
    return loaded
