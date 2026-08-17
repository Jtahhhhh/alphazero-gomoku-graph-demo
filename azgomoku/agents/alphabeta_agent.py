"""Alpha-beta depth-limited agent for Gomoku."""

import numpy as np


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


class AlphaBetaAgent:
    """
    Depth-limited alpha-beta search agent.
    
    Does NOT use exact solver for 15x15 boards (computationally infeasible).
    Uses heuristic evaluation instead.
    
    Args:
        depth: Search depth (typically 4, 6, or 8)
        board_size: Board size (default 15 for arena)
        win_length: Win condition length (default 5 for gomoku)
    """

    def __init__(self, depth=4, board_size=15, win_length=5):
        self.depth = depth
        self.board_size = board_size
        self.win_length = win_length
        self.nodes_explored = 0

    def select_move(self, state):
        """
        Select best move using alpha-beta search.
        
        Args:
            state: GameState object
            
        Returns:
            action: Integer index of selected move
        """
        self.nodes_explored = 0
        legal_moves = state.legal_actions()

        if len(legal_moves) == 0:
            return None

        # Opening move heuristic: play in center
        if len(legal_moves) == self.board_size * self.board_size:
            center = self.board_size // 2
            return center * self.board_size + center

        # Immediate tactical win should always take precedence.
        for move in legal_moves:
            next_state = state.play(move)
            if next_state.winner() == state.to_play:
                return int(move)

        # If the opponent has an immediate winning move after our move,
        # reject that move and choose a response that removes all tactical threats.
        valid_block_moves = []
        for move in legal_moves:
            next_state = state.play(move)
            opponent_can_win_after = False
            for opp_move in next_state.legal_actions():
                opp_state = next_state.play(opp_move)
                if opp_state.winner() == -state.to_play:
                    opponent_can_win_after = True
                    break
            if not opponent_can_win_after:
                valid_block_moves.append(int(move))

        if valid_block_moves:
            return int(valid_block_moves[0])

        best_score = float("-inf")
        best_move = int(legal_moves[0])

        for move in legal_moves:
            next_state = state.play(move)
            score = self._alpha_beta(next_state, self.depth - 1, float("-inf"), float("inf"), False)
            if score > best_score:
                best_score = score
                best_move = int(move)

        return best_move

    def _alpha_beta(self, state, depth, alpha, beta, is_maximizing):
        """
        Alpha-beta search with depth limit.
        
        Args:
            state: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: True if maximizing player (current player), False otherwise
            
        Returns:
            Heuristic score for this state
        """
        self.nodes_explored += 1

        # Terminal conditions
        winner = state.winner()
        if winner == state.to_play:
            return 1.0  # Current player wins
        elif winner == -state.to_play:
            return -1.0  # Opponent wins
        elif winner == 0 and len(state.legal_actions()) == 0:
            return 0.0  # Draw
        elif depth == 0:
            return evaluate_position(state, self.board_size, self.win_length)

        legal_moves = self.get_ordered_moves(state)

        if is_maximizing:
            max_eval = float("-inf")
            for move in legal_moves:
                next_state = state.play(move)
                eval = self._alpha_beta(next_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:
            min_eval = float("inf")
            for move in legal_moves:
                next_state = state.play(move)
                eval = self._alpha_beta(next_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval

    def get_ordered_moves(self, state):
        """
        Return legal moves ordered by heuristic quality.
        
        Orders moves by proximity to existing stones to reduce branching factor.
        
        Args:
            state: Current game state
            
        Returns:
            List of moves ordered by heuristic priority
        """
        legal_moves = state.legal_actions()

        if len(legal_moves) == 0:
            return []

        board = state.board
        board_size = state.size

        # Score each move based on proximity to existing stones
        move_scores = []
        for move in legal_moves:
            row, col = divmod(move, board_size)
            
            # Find minimum distance to any existing stone
            min_dist = float("inf")
            for r in range(board_size):
                for c in range(board_size):
                    if board[r, c] != 0:
                        dist = abs(row - r) + abs(col - c)
                        min_dist = min(min_dist, dist)
            
            # Moves closer to existing stones have higher priority
            priority = -min_dist if min_dist != float("inf") else 0
            move_scores.append((priority, move))

        # Sort by priority (descending) and return moves
        move_scores.sort(reverse=True)
        return [move for _, move in move_scores]
