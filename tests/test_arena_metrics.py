import csv
import json

import torch
from torch import nn

from investigation.arena import arena_score, delta_elo, run_arena, summarize


def row(model, opponent, result, side="black"):
    return {"model": model, "opponent": opponent, "result": result, "model_side": side}


def test_arena_score_counts_draw_as_half():
    assert arena_score(610, 120, 270) == 0.67
    assert delta_elo(0.67, 1000) > 120


def test_summarize_keeps_first_second_scores_and_pairwise_elo():
    rows = [row("rgat", "egreedy", "win", "black"), row("rgat", "egreedy", "draw", "black"), row("rgat", "egreedy", "loss", "white")]
    summaries, matchups = summarize(rows)
    assert summaries[0]["games"] == 3
    assert summaries[0]["first_score"] == 0.75
    assert summaries[0]["second_score"] == 0.0
    assert matchups[0]["score"] == 0.5


class FlatModel(nn.Module):
    def __init__(self, board_size=3):
        super().__init__()
        self.board_size = board_size
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, return_evidence=False):
        batch = x.shape[0]
        logits = self.bias.expand(batch, self.board_size * self.board_size)
        values = self.bias.expand(batch)
        return logits, values


def test_run_arena_writes_raw_summary_and_global_elo(tmp_path):
    run_arena({"flat": FlatModel()}, tmp_path, games=2, board_size=3, win_length=3, mcts_playouts=1, depth=1)
    with (tmp_path / "arena_games.csv").open(newline="", encoding="utf-8") as handle:
        games = list(csv.DictReader(handle))
    with (tmp_path / "arena_summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    elo = json.loads((tmp_path / "arena_elo.json").read_text(encoding="utf-8"))
    assert len(games) == 4
    assert all(row["model_side"] in {"black", "white"} for row in games)
    assert {row["model_side"] for row in games} == {"black", "white"}
    assert len(summary) == 2
    assert {"first_score", "second_score", "delta_elo"} <= set(summary[0])
    assert elo["reference"] == {"agent": "egreedy", "elo": 0.0}
