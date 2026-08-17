"""Epsilon-greedy heuristic agent for Gomoku."""

import numpy as np

from azgomoku.tactics import mandatory_defenses


def _preferred_block_move(state, candidates):
    """Pick a deterministic block among multiple immediate threat cells."""

    candidate_list = [int(move) for move in candidates]
    if not candidate_list:
        return None
    if len(candidate_list) == 1:
        return candidate_list[0]

    center = (state.size - 1) / 2.0
    return min(
        candidate_list,
        key=lambda move: (
            abs(move // state.size - center) + abs(move % state.size - center),
            -move,
        ),
    )


def evaluate_position(state, board_size=15, win_length=5):
    """
    Heuristic evaluation of board state from perspective of current player.
    
    Scoring based on:
    - Threat patterns (open-three, open-four, etc.)
    - Attacked patterns (blocked sequences)
    - Center control bonus
    - Basic connectivity
    
    Returns: score in range [-1, 1] where 1 = strong for current player.
    """
    board = state.board
    board_size = state.size
    to_play = state.to_play
    opponent = -to_play

    # Check for immediate win/loss
    # If current player can win in 1 move
    for action in state.legal_actions():
        next_state = state.play(action)
        if next_state.winner() == to_play:
            return 1.0  # Winning move

    # If opponent can win in 1 move (block it)
    for action in state.legal_actions():
        # Simulate opponent move
        board_copy = state.board.copy()
        board_copy.reshape(-1)[action] = opponent
        from azgomoku.game import GomokuState
        test_state = GomokuState(board=board_copy, to_play=opponent, last_move=action, win_length=state.win_length)
        if test_state.winner() == opponent:
            return -0.95  # Critical defense needed

    score = 0.0

    # Scan for patterns
    directions = [
        (0, 1),  # horizontal
        (1, 0),  # vertical
        (1, 1),  # diagonal
        (1, -1),  # anti-diagonal
    ]

    for row in range(board_size):
        for col in range(board_size):
            if board[row, col] != 0:
                continue  # Skip occupied positions

            # Evaluate placing current player's stone here
            for dr, dc in directions:
                # Count consecutive player stones in this direction
                for player in [to_play, opponent]:
                    same_dir = 0
                    opp_dir = 0

                    # Count forward
                    r, c = row + dr, col + dc
                    while 0 <= r < board_size and 0 <= c < board_size:
                        if board[r, c] == player:
                            same_dir += 1
                        elif board[r, c] == 0:
                            break
                        else:
                            break
                        r, c = r + dr, c + dc

                    # Count backward
                    r, c = row - dr, col - dc
                    while 0 <= r < board_size and 0 <= c < board_size:
                        if board[r, c] == player:
                            opp_dir += 1
                        elif board[r, c] == 0:
                            break
                        else:
                            break
                        r, c = r - dr, c - dc

                    total = same_dir + opp_dir
                    is_own_move = player == to_play

                    # Scoring based on length
                    if total >= 4:
                        score += 0.8 if is_own_move else -0.7
                    elif total == 3:
                        score += 0.6 if is_own_move else -0.5
                    elif total == 2:
                        score += 0.3 if is_own_move else -0.3
                    elif total == 1:
                        score += 0.1 if is_own_move else -0.1

            # Center control bonus (closer to center = better for exploration)
            dist_to_center = (
                abs(row - board_size / 2.0) + abs(col - board_size / 2.0)
            ) / board_size
            center_bonus = (0.5 - dist_to_center) * 0.05
            score += center_bonus

    # Normalize score to [-1, 1]
    score = np.clip(score / 10.0, -1.0, 1.0)
    return float(score)


class EGreedyAgent:
    """
    Epsilon-greedy agent using 1-ply heuristic evaluation.
    
    Evaluates all legal moves with heuristic, selects best with probability (1-epsilon),
    selects random with probability epsilon.
    
    Args:
        epsilon: Exploration rate (0.05, 0.1, or 0.2 typical)
        board_size: Board size (default 15 for arena)
        win_length: Win condition length (default 5 for gomoku)
        seed: Random seed for reproducibility (optional)
    """

    def __init__(self, epsilon=0.1, board_size=15, win_length=5, seed=None):
        self.epsilon = epsilon
        self.board_size = board_size
        self.win_length = win_length
        self.rng = np.random.RandomState(seed)

    def select_move(self, state):
        """
        Select move using epsilon-greedy 1-ply heuristic.
        
        Args:
            state: GameState object
            
        Returns:
            action: Integer index of selected move
        """
        legal_moves = state.legal_actions()

        if len(legal_moves) == 0:
            return None

        # Immediate tactical wins should always be taken before exploring.
        for move in legal_moves:
            next_state = state.play(move)
            if next_state.winner() == state.to_play:
                return int(move)

        # If the opponent has a forced immediate win, blocking is mandatory.
        defense = mandatory_defenses(state, -state.to_play)
        if defense.completions:
            return _preferred_block_move(
                state, defense.blocking_moves or defense.completions
            )

        # Exploration: with probability epsilon, select random move
        if self.rng.rand() < self.epsilon:
            return int(self.rng.choice(legal_moves))

        # Exploitation: select move with highest 1-ply heuristic evaluation
        best_score = float("-inf")
        best_moves = []

        for move in legal_moves:
            next_state = state.play(move)
            score = -evaluate_position(next_state, self.board_size, self.win_length)

            if score > best_score:
                best_score = score
                best_moves = [int(move)]
            elif score == best_score:
                best_moves.append(int(move))

        # If multiple moves tied, select randomly among them
        return int(self.rng.choice(best_moves)) if best_moves else int(legal_moves[0])
