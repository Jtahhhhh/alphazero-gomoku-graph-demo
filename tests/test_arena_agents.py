"""Tests for arena components: agents, eval harness, and match tracking."""

import pytest
import numpy as np
from azgomoku.game import GomokuState
from azgomoku.agents.alphabeta_agent import AlphaBetaAgent
from azgomoku.agents.egreedy_agent import EGreedyAgent
from azgomoku.tracking.match_tracker import check_milestone


class TestCheckMilestone:
    """Tests for milestone detection logic."""

    def test_check_milestone_reached(self):
        """Test milestone detection when threshold is reached."""
        # 3 consecutive evals at 95%+
        win_rates = [0.80, 0.85, 0.95, 0.96, 0.97]
        games_per_eval = 40

        result = check_milestone(win_rates, games_per_eval, threshold=0.95, min_consecutive=3)
        assert result == 2, "Should detect milestone at index 2"

    def test_check_milestone_not_enough_consecutive(self):
        """Test when only 1-2 evals reach threshold (not enough)."""
        win_rates = [0.80, 0.85, 0.90, 0.96]
        games_per_eval = 40

        result = check_milestone(win_rates, games_per_eval, threshold=0.95, min_consecutive=3)
        assert result is None, "Should not detect milestone with only 1 eval above threshold"

    def test_check_milestone_drops_below(self):
        """Test when win rate reaches threshold but drops below later."""
        win_rates = [0.85, 0.96, 0.97, 0.98, 0.60, 0.65]
        games_per_eval = 40

        result = check_milestone(win_rates, games_per_eval, threshold=0.95, min_consecutive=3)
        assert result == 1, "Should detect milestone even if it drops later"

    def test_check_milestone_insufficient_data(self):
        """Test with not enough data points."""
        win_rates = [0.95, 0.96]
        games_per_eval = 40

        result = check_milestone(win_rates, games_per_eval, threshold=0.95, min_consecutive=3)
        assert result is None, "Should return None with insufficient data"


class TestAlphaBetaAgent:
    """Tests for alpha-beta agent."""

    def test_alphabeta_immediate_win(self):
        """Test that alpha-beta takes immediate win."""
        # Create position where alpha-beta can win in 1 move
        state = GomokuState.initial(size=6, win_length=4)
        board = state.board.copy()
        board[2, 0:3] = 1
        board[2, 3] = 0  # Empty spot for win
        state = GomokuState(board=board, to_play=1, last_move=-1, win_length=4)

        agent = AlphaBetaAgent(depth=2, board_size=6, win_length=4)
        move = agent.select_move(state)

        # Should choose the winning move at position [2, 3]
        expected_move = 2 * 6 + 3  # Row 2, Col 3
        assert move == expected_move, f"Should select winning move {expected_move}, got {move}"

    def test_alphabeta_blocks_opponent_win(self):
        """Test that alpha-beta blocks opponent's winning move."""
        state = GomokuState.initial(size=6, win_length=4)
        board = state.board.copy()
        board[1, 1:4] = -1
        board[1, 4] = 0  # Empty spot for opponent to win
        state = GomokuState(board=board, to_play=1, last_move=-1, win_length=4)

        agent = AlphaBetaAgent(depth=2, board_size=6, win_length=4)
        move = agent.select_move(state)

        # Should choose the blocking move
        expected_move = 1 * 6 + 4  # Row 1, Col 4
        assert move == expected_move, f"Should block opponent winning move at {expected_move}, got {move}"

    def test_alphabeta_opening_move(self):
        """Test that alpha-beta plays center opening on empty board."""
        state = GomokuState.initial(size=6, win_length=4)

        agent = AlphaBetaAgent(depth=2, board_size=6, win_length=4)
        move = agent.select_move(state)

        # Should play in center (3,3) for 6x6 board
        expected_move = 3 * 6 + 3
        assert move == expected_move, f"Should play center opening at {expected_move}, got {move}"


class TestEGreedyAgent:
    """Tests for epsilon-greedy agent."""

    def test_egreedy_takes_win(self):
        """Test that epsilon-greedy takes immediate win even with high epsilon."""
        state = GomokuState.initial(size=6, win_length=4)
        board = state.board.copy()
        board[2, 0:3] = 1
        board[2, 3] = 0
        state = GomokuState(board=board, to_play=1, last_move=-1, win_length=4)

        # Even with high epsilon, should take winning move
        agent = EGreedyAgent(epsilon=0.9, board_size=6, win_length=4, seed=42)
        move = agent.select_move(state)

        expected_move = 2 * 6 + 3
        assert move == expected_move, "Should always take immediate win"

    def test_egreedy_blocks_opponent(self):
        """Test that epsilon-greedy blocks opponent's winning move with high priority."""
        state = GomokuState.initial(size=6, win_length=4)
        board = state.board.copy()
        board[1, 1:4] = -1
        board[1, 4] = 0
        state = GomokuState(board=board, to_play=1, last_move=-1, win_length=4)

        agent = EGreedyAgent(epsilon=0.95, board_size=6, win_length=4, seed=42)
        move = agent.select_move(state)

        expected_move = 1 * 6 + 4
        assert move == expected_move, "Should prioritize blocking opponent win"

    def test_egreedy_exploration(self):
        """Test that epsilon-greedy explores with some probability."""
        state = GomokuState.initial(size=6, win_length=4)
        board = state.board.copy()
        board[2, 2] = 1
        board[2, 3] = -1
        state = GomokuState(board=board, to_play=1, last_move=-1, win_length=4)

        agent = EGreedyAgent(epsilon=1.0, board_size=6, win_length=4, seed=42)

        # With epsilon=1.0, should always explore (random)
        moves = set()
        for _ in range(10):
            move = agent.select_move(state)
            moves.add(move)

        # Should have some variety in moves with high epsilon and random seed
        assert len(moves) > 1, "Should explore with epsilon=1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
