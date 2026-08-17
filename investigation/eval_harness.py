"""Standardized evaluation harness for model benchmarking against heuristic opponents."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json
import numpy as np

from azgomoku.game import GomokuState
from azgomoku.agents.alphabeta_agent import AlphaBetaAgent
from azgomoku.agents.egreedy_agent import EGreedyAgent
from azgomoku.tracking.match_tracker import GameRecord, JsonGameLogger
import torch


@dataclass
class EvalResult:
    """Result from evaluating a model against one opponent."""

    model_name: str
    iteration: int
    opponent_name: str
    opponent_strength: float  # epsilon or depth
    n_games: int
    wins: int
    draws: int
    losses: int
    duration_seconds: float

    @property
    def win_rate(self) -> float:
        """Win rate: wins / n_games."""
        return self.wins / self.n_games if self.n_games > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "iteration": self.iteration,
            "opponent_name": self.opponent_name,
            "opponent_strength": self.opponent_strength,
            "n_games": self.n_games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "duration_seconds": self.duration_seconds,
        }


def create_opponent_agent(opponent_type: str, strength: float, board_size: int = 15, win_length: int = 5):
    """
    Create an opponent agent.

    Args:
        opponent_type: "egreedy" or "alphabeta"
        strength: epsilon for egreedy, depth for alphabeta
        board_size: Board size
        win_length: Win condition length

    Returns:
        Agent instance
    """
    if opponent_type == "egreedy":
        return EGreedyAgent(
            epsilon=strength,
            board_size=board_size,
            win_length=win_length,
        )
    elif opponent_type == "alphabeta":
        return AlphaBetaAgent(
            depth=int(strength),
            board_size=board_size,
            win_length=win_length,
        )
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")


def run_game(
    model,
    opponent,
    board_size: int = 15,
    win_length: int = 5,
    mcts_playouts: int = 400,
    model_plays_first: bool = True,
):
    """
    Run a single game between model (using MCTS) and opponent.

    Args:
        model: Neural network model (CNN, RGCN, or RGAT)
        opponent: Opponent agent instance
        board_size: Board size
        win_length: Win condition
        mcts_playouts: Number of MCTS playouts for model
        model_plays_first: If True, model plays as player 1 (first)

    Returns:
        Tuple of (winner, n_moves, duration_seconds)
        winner: 1 if model wins, -1 if opponent wins, 0 if draw
    """
    from azgomoku.mcts import search

    start_time = time.perf_counter()
    state = GomokuState.initial(size=board_size, win_length=win_length)

    if not model_plays_first:
        # Opponent plays first
        state = GomokuState(board=state.board.copy(), to_play=-1, last_move=-1, win_length=win_length)

    move_count = 0

    while state.winner() == 0 and len(state.legal_actions()) > 0:
        if (state.to_play == 1) == model_plays_first:
            # Model's turn - use MCTS search
            pi = search(model, state, playouts=mcts_playouts, temperature=1.0)
            action = int(np.argmax(pi))
        else:
            # Opponent's turn
            action = opponent.select_move(state)

        if action is None:
            break

        state = state.play(action)
        move_count += 1

    duration = time.perf_counter() - start_time

    winner = state.winner()
    model_result = 1 if winner == (1 if model_plays_first else -1) else (-1 if winner != 0 else 0)

    return model_result, move_count, duration


def run_evaluation(
    model,
    model_name: str,
    iteration: int,
    opponents: List[tuple],  # List of (opponent_type, strength) tuples
    n_games_per_opponent: int = 40,
    mcts_playouts_eval: int = 400,
    board_size: int = 15,
    win_length: int = 5,
    log_dir: Optional[Path] = None,
) -> List[EvalResult]:
    """
    Run evaluation of model against multiple opponents.

    Args:
        model: Neural network model
        model_name: Name of model for logging
        iteration: Training iteration number
        opponents: List of (opponent_type, strength) tuples
        n_games_per_opponent: Games to play against each opponent (split between first/second)
        mcts_playouts_eval: MCTS playouts during evaluation
        board_size: Board size
        win_length: Win condition
        log_dir: Directory to store eval logs (optional)

    Returns:
        List of EvalResult objects
    """
    results = []
    device = next(model.parameters()).device
    model.eval()

    logger = JsonGameLogger(log_dir) if log_dir else None

    with torch.no_grad():
        for opponent_type, strength in opponents:
            games_as_first = n_games_per_opponent // 2
            games_as_second = n_games_per_opponent - games_as_first

            wins = draws = losses = 0
            total_duration = 0.0
            game_count = 0

            # Games where model plays first
            for game_num in range(games_as_first):
                opponent = create_opponent_agent(
                    opponent_type, strength, board_size, win_length
                )
                result, n_moves, duration = run_game(
                    model,
                    opponent,
                    board_size=board_size,
                    win_length=win_length,
                    mcts_playouts=mcts_playouts_eval,
                    model_plays_first=True,
                )

                if result == 1:
                    wins += 1
                    result_str = "win"
                elif result == 0:
                    draws += 1
                    result_str = "draw"
                else:
                    losses += 1
                    result_str = "loss"

                total_duration += duration
                game_count += 1

                if logger:
                    record = GameRecord(
                        game_id=f"{model_name}_iter{iteration}_opp{opponent_type}{strength}_game{game_count}",
                        timestamp=datetime.now().isoformat(),
                        model_name=model_name,
                        iteration=iteration,
                        opponent_type=opponent_type,
                        opponent_strength=strength,
                        model_plays_first=True,
                        result=result_str,
                        n_moves=n_moves,
                        duration_seconds=duration,
                    )
                    logger.log_game(record)

            # Games where model plays second
            for game_num in range(games_as_second):
                opponent = create_opponent_agent(
                    opponent_type, strength, board_size, win_length
                )
                result, n_moves, duration = run_game(
                    model,
                    opponent,
                    board_size=board_size,
                    win_length=win_length,
                    mcts_playouts=mcts_playouts_eval,
                    model_plays_first=False,
                )

                if result == 1:
                    wins += 1
                    result_str = "win"
                elif result == 0:
                    draws += 1
                    result_str = "draw"
                else:
                    losses += 1
                    result_str = "loss"

                total_duration += duration
                game_count += 1

                if logger:
                    record = GameRecord(
                        game_id=f"{model_name}_iter{iteration}_opp{opponent_type}{strength}_game{game_count}",
                        timestamp=datetime.now().isoformat(),
                        model_name=model_name,
                        iteration=iteration,
                        opponent_type=opponent_type,
                        opponent_strength=strength,
                        model_plays_first=False,
                        result=result_str,
                        n_moves=n_moves,
                        duration_seconds=duration,
                    )
                    logger.log_game(record)

            # Create result object
            result = EvalResult(
                model_name=model_name,
                iteration=iteration,
                opponent_name=opponent_type,
                opponent_strength=strength,
                n_games=game_count,
                wins=wins,
                draws=draws,
                losses=losses,
                duration_seconds=total_duration,
            )
            results.append(result)

    return results
