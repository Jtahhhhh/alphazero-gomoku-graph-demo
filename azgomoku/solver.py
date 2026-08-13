"""Bounded exact Gomoku solver following the H1 player-to-move contract."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Literal

from .game import GomokuState


Status = Literal["exact", "timeout", "node_budget"]


@dataclass(frozen=True)
class SolverResult:
    status: Status
    value: int | None
    optimal_actions: tuple[int, ...]
    action_values: dict[int, int]
    nodes: int
    elapsed_ms: float

    def dict(self):
        result = asdict(self)
        result["optimal_actions"] = list(self.optimal_actions)
        result["action_values"] = {str(k): v for k, v in self.action_values.items()}
        return result


class _LimitReached(RuntimeError):
    def __init__(self, status: Status): self.status = status


class _Context:
    def __init__(self, deadline_ms, node_budget):
        self.start = time.perf_counter()
        self.deadline = None if deadline_ms is None else self.start + deadline_ms / 1000
        self.node_budget = node_budget
        self.nodes = 0
        self.table: dict[tuple[bytes, int, int], int] = {}

    def enter(self):
        if self.deadline is not None and time.perf_counter() >= self.deadline: raise _LimitReached("timeout")
        if self.node_budget is not None and self.nodes >= self.node_budget: raise _LimitReached("node_budget")
        self.nodes += 1


def _ordered_actions(state: GomokuState):
    actions = [int(a) for a in state.legal_actions()]
    wins = [a for a in actions if state.play(a).winner() == state.to_play]
    winning = set(wins); center = (state.size - 1) / 2
    rest = [a for a in actions if a not in winning]
    rest.sort(key=lambda a: (abs(a // state.size - center) + abs(a % state.size - center), a))
    return wins + rest


def _negamax(state: GomokuState, alpha: int, beta: int, ctx: _Context) -> int:
    ctx.enter()
    if state.terminal(): return state.outcome_for(state.to_play)
    key = (state.board.tobytes(), int(state.to_play), int(state.win_length))
    if key in ctx.table: return ctx.table[key]
    best = -2; cutoff = False
    for action in _ordered_actions(state):
        value = -_negamax(state.play(action), -beta, -alpha, ctx)
        best = max(best, value); alpha = max(alpha, value)
        if alpha >= beta:
            cutoff = True
            break
    # A cutoff returns a bound, not necessarily an exact value; do not cache it as exact.
    if not cutoff: ctx.table[key] = best
    return best


def solve_actions(state: GomokuState, deadline_ms=None, node_budget=None) -> SolverResult:
    ctx = _Context(deadline_ms, node_budget); values: dict[int, int] = {}
    status: Status = "exact"
    try:
        if state.terminal():
            value = state.outcome_for(state.to_play)
            return SolverResult("exact", value, (), {}, 0, (time.perf_counter()-ctx.start)*1000)
        for action in _ordered_actions(state):
            # Full window per root action guarantees every reported action value is exact.
            values[action] = -_negamax(state.play(action), -1, 1, ctx)
    except _LimitReached as exc:
        status = exc.status
    elapsed = (time.perf_counter() - ctx.start) * 1000
    if status != "exact": return SolverResult(status, None, (), values, ctx.nodes, elapsed)
    value = max(values.values()) if values else state.outcome_for(state.to_play)
    optimal = tuple(sorted(action for action, candidate in values.items() if candidate == value))
    return SolverResult("exact", value, optimal, values, ctx.nodes, elapsed)


def solve_state(state: GomokuState, deadline_ms=None, node_budget=None) -> SolverResult:
    return solve_actions(state, deadline_ms=deadline_ms, node_budget=node_budget)
