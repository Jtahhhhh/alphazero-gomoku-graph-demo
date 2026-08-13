"""Geometry-certified tactical proofs, deliberately independent of the solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .game import GomokuState
from .graph import RELATIONS


DIRECTIONS = tuple(zip(RELATIONS, ((0, 1), (1, 0), (1, 1), (1, -1))))


@dataclass(frozen=True, order=True)
class WinningThreat:
    """One legal empty cell that completes a concrete winning window."""

    completion: int
    relation: str
    window: tuple[int, ...]


@dataclass(frozen=True)
class DefenseSet:
    """Sound blocking summary for immediate winning threats only."""

    completions: tuple[int, ...]
    blocking_moves: tuple[int, ...]
    unstoppable: bool


@dataclass(frozen=True)
class MoveThreats:
    """Threat primitives created by one legal move; not a win certificate."""

    move: int
    creates_five: bool
    fours: tuple[WinningThreat, ...]
    three_extensions: tuple[int, ...]

    @property
    def creates_four(self) -> bool:
        return bool(self.fours)

    @property
    def creates_three(self) -> bool:
        return bool(self.three_extensions)

    @property
    def creates_open_three(self) -> bool:
        return len(self.three_extensions) >= 2

    @property
    def creates_double_four(self) -> bool:
        return len({threat.completion for threat in self.fours}) >= 2

    @property
    def creates_four_three(self) -> bool:
        return self.creates_four and self.creates_three


def windows(size: int, length: int):
    for relation, (dr, dc) in DIRECTIONS:
        for row in range(size):
            for col in range(size):
                end_row, end_col = row + (length-1)*dr, col + (length-1)*dc
                if 0 <= end_row < size and 0 <= end_col < size:
                    yield relation, tuple((row+i*dr)*size + col+i*dc for i in range(length))


def immediate_threats(board: np.ndarray, player: int, win_length: int):
    flat = board.reshape(-1); found = []
    for relation, cells in windows(board.shape[0], win_length):
        values = [int(flat[cell]) for cell in cells]
        empties = [cell for cell, value in zip(cells, values) if value == 0]
        if len(empties) == 1 and values.count(player) == win_length-1:
            found.append((empties[0], relation, cells))
    return found


def _state_for_player(state: GomokuState, player: int) -> GomokuState:
    if player not in (-1, 1):
        raise ValueError("player must be -1 or 1")
    return GomokuState(
        state.board,
        to_play=player,
        last_move=state.last_move,
        win_length=state.win_length,
    )


def creates_five(state: GomokuState, move: int, player: int) -> bool:
    """Return whether a legal placement by ``player`` wins immediately."""

    player_state = _state_for_player(state, player)
    if int(move) not in set(map(int, player_state.legal_actions())):
        return False
    return player_state.play(int(move)).winner() == player


def winning_completions(state: GomokuState, player: int) -> tuple[WinningThreat, ...]:
    """Enumerate and replay-verify all immediate winning windows."""

    threats = []
    for completion, relation, window in immediate_threats(
        state.board, player, state.win_length
    ):
        if creates_five(state, completion, player):
            threats.append(WinningThreat(int(completion), relation, tuple(window)))
    return tuple(sorted(set(threats)))


def mandatory_defenses(state: GomokuState, attacker: int) -> DefenseSet:
    """Return blocks for current immediate wins, without heuristic pruning.

    A placement can occupy only one completion cell. Therefore multiple distinct
    completion cells are unstoppable by a single ordinary blocking move. Counter-
    wins are deliberately left to the future AND-node logic.
    """

    completions = tuple(sorted({t.completion for t in winning_completions(state, attacker)}))
    return DefenseSet(
        completions=completions,
        blocking_moves=completions if len(completions) == 1 else (),
        unstoppable=len(completions) >= 2,
    )


def _fours_after_move(state: GomokuState, move: int, player: int) -> tuple[WinningThreat, ...]:
    player_state = _state_for_player(state, player)
    if int(move) not in set(map(int, player_state.legal_actions())):
        return ()
    child = player_state.play(int(move))
    if child.winner() == player:
        return ()
    return winning_completions(child, player)


def three_extensions(state: GomokuState, move: int, player: int) -> tuple[int, ...]:
    """Legal next attacker moves that would create a four after ``move``.

    This is intentionally a candidate primitive. It does not claim that the same
    extension survives every intervening defender reply.
    """

    player_state = _state_for_player(state, player)
    if int(move) not in set(map(int, player_state.legal_actions())):
        return ()
    child = player_state.play(int(move))
    if child.winner() == player:
        return ()
    extensions = []
    for extension in map(int, child.legal_actions()):
        if _fours_after_move(child, extension, player):
            extensions.append(extension)
    return tuple(sorted(extensions))


def classify_threat_move(state: GomokuState, move: int, player: int) -> MoveThreats:
    """Classify one move using replay-verified five/four/three primitives."""

    return MoveThreats(
        move=int(move),
        creates_five=creates_five(state, move, player),
        fours=_fours_after_move(state, move, player),
        three_extensions=three_extensions(state, move, player),
    )


def threat_moves(
    state: GomokuState, player: int, method: Literal["vcf", "vct"]
) -> tuple[MoveThreats, ...]:
    """Deterministically enumerate attacker candidates for a future solver."""

    if method not in ("vcf", "vct"):
        raise ValueError("method must be 'vcf' or 'vct'")
    candidates = []
    for move in map(int, state.legal_actions()):
        # VCF must neither admit nor pay to enumerate three candidates.
        if method == "vcf":
            classified = MoveThreats(
                move=move,
                creates_five=creates_five(state, move, player),
                fours=_fours_after_move(state, move, player),
                three_extensions=(),
            )
        else:
            classified = classify_threat_move(state, move, player)
        if classified.creates_five or classified.creates_four:
            candidates.append(classified)
        elif method == "vct" and classified.creates_three:
            candidates.append(classified)
    return tuple(candidates)


def extract_tactical_proofs(state: GomokuState):
    proofs = []
    opponent_threats = immediate_threats(state.board, -state.to_play, state.win_length)
    blocks: dict[int, list[tuple[str, tuple[int, ...]]]] = {}
    # A block is mandatory only when one legal cell answers every immediate threat.
    threat_cells={action for action,_,_ in opponent_threats}
    if len(threat_cells)==1:
        for action, relation, cells in opponent_threats: blocks.setdefault(action, []).append((relation, cells))
    for action in map(int, state.legal_actions()):
        child = state.play(action)
        wins = []
        if child.winner() == state.to_play:
            flat = child.board.reshape(-1)
            for relation, cells in windows(state.size, state.win_length):
                if action in cells and all(flat[cell] == state.to_play for cell in cells): wins.append((relation, cells))
        for relation, cells in wins:
            proofs.append({"action":action,"concepts":["immediate_win","winning_line"],"critical_cells":list(cells),"critical_relations":[relation],"windows":[list(cells)]})
        if action in blocks:
            entries = blocks[action]
            proofs.append({"action":action,"concepts":["mandatory_block"],"critical_cells":sorted({cell for _,cells in entries for cell in cells}),"critical_relations":sorted({relation for relation,_ in entries}),"windows":[list(cells) for _,cells in entries]})
        if not child.terminal():
            threats = immediate_threats(child.board, state.to_play, state.win_length)
            reply_cells = {empty for empty,_,_ in threats}
            if len(reply_cells) >= 2:
                proofs.append({"action":action,"concepts":["simple_fork"],"critical_cells":sorted({action}|{cell for _,_,cells in threats for cell in cells}),"critical_relations":sorted({relation for _,relation,_ in threats}),"windows":[list(cells) for _,_,cells in threats]})
    proofs.sort(key=lambda p:(p["action"],p["concepts"],p["critical_relations"],p["windows"]))
    return proofs
